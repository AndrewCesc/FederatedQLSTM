import matplotlib.pyplot as plt
import pickle
from datetime import datetime
import time, os, copy

import torch
import torch.nn as nn
import torch.optim as optim

# Pennylane e numpy interno
import pennylane as qml
from pennylane import numpy as np

# Importa eventuali moduli custom per i circuiti quantistici
from metaquantum.CircuitComponents import VQCVariationalLoadingFlexNoisy
from metaquantum import Optimization

import qiskit
import qiskit.providers.aer.noise as noise

# =============================================================================
# Classe VQLSTM – Layer ibrido LSTM con componente quantistica
# Modifica: ora il metodo forward accetta il flag "return_all" per poter
# restituire l'intera sequenza (utile nel decoder) oppure solo l'ultimo output
# (utile nell'encoder, per ottenere la rappresentazione compressa).
# =============================================================================
class VQLSTM(nn.Module):
    def __init__(self, 
                 lstm_input_size, 
                 lstm_hidden_size,
                 lstm_output_size,
                 lstm_num_qubit,
                 lstm_cell_cat_size,
                 lstm_cell_num_layers,
                 lstm_internal_size,
                 duplicate_time_of_input,
                 as_reservoir,
                 single_y,
                 output_all_h,
                 qdevice,
                 dev,
                 gpu_q):
        super().__init__()

        # Parametri del layer
        self.lstm_input_size = lstm_input_size
        self.lstm_hidden_size = lstm_hidden_size
        self.lstm_output_size = lstm_output_size
        self.lstm_num_qubit = lstm_num_qubit
        self.lstm_cell_cat_size = lstm_cell_cat_size
        self.lstm_cell_num_layers = lstm_cell_num_layers
        self.lstm_internal_size = lstm_internal_size
        self.duplicate_time_of_input = duplicate_time_of_input
        self.as_reservoir = as_reservoir
        self.single_y = single_y
        self.output_all_h = output_all_h

        self.qdevice = qdevice
        self.dev = dev
        self.gpu_q = gpu_q

        # Parametri quantistici per ogni “cella”
        self.q_params_1 = nn.Parameter(0.01 * torch.randn(lstm_cell_num_layers, lstm_num_qubit, 3))
        self.q_params_2 = nn.Parameter(0.01 * torch.randn(lstm_cell_num_layers, lstm_num_qubit, 3))
        self.q_params_3 = nn.Parameter(0.01 * torch.randn(lstm_cell_num_layers, lstm_num_qubit, 3))
        self.q_params_4 = nn.Parameter(0.01 * torch.randn(lstm_cell_num_layers, lstm_num_qubit, 3))
        self.q_params_5 = nn.Parameter(0.01 * torch.randn(lstm_cell_num_layers, lstm_num_qubit, 3))
        self.q_params_6 = nn.Parameter(0.01 * torch.randn(lstm_cell_num_layers, lstm_num_qubit, 3))

        if self.as_reservoir:
            self.q_params_1.requires_grad = False
            self.q_params_2.requires_grad = False
            self.q_params_3.requires_grad = False
            self.q_params_4.requires_grad = False
            self.q_params_5.requires_grad = False
            self.q_params_6.requires_grad = False

        # Rete neurale classica (per trasformare l'output quantistico in un valore finale)
        self.classical_nn_linear = nn.Linear(lstm_output_size, 1)

        # Definizione delle “celle” quantistiche (i blocchi operativi)
        self.cell_1 = VQCVariationalLoadingFlexNoisy(
            num_of_input= lstm_cell_cat_size,
            num_of_output= lstm_internal_size,
            num_of_wires = 4,
            num_of_layers = lstm_cell_num_layers,
            qdevice = qdevice,
            hadamard_gate = True,
            more_entangle = True,
            gpu = gpu_q,
            noisy_dev = dev)
        self.cell_2 = VQCVariationalLoadingFlexNoisy(
            num_of_input= lstm_cell_cat_size,
            num_of_output= lstm_internal_size,
            num_of_wires = 4,
            num_of_layers = lstm_cell_num_layers,
            qdevice = qdevice,
            hadamard_gate = True,
            more_entangle = True,
            gpu = gpu_q,
            noisy_dev = dev)
        self.cell_3 = VQCVariationalLoadingFlexNoisy(
            num_of_input= lstm_cell_cat_size,
            num_of_output= lstm_internal_size,
            num_of_wires = 4,
            num_of_layers = lstm_cell_num_layers,
            qdevice = qdevice,
            hadamard_gate = True,
            more_entangle = True,
            gpu = gpu_q,
            noisy_dev = dev)
        self.cell_4 = VQCVariationalLoadingFlexNoisy(
            num_of_input= lstm_cell_cat_size,
            num_of_output= lstm_internal_size,
            num_of_wires = 4,
            num_of_layers = lstm_cell_num_layers,
            qdevice = qdevice,
            hadamard_gate = True,
            more_entangle = True,
            gpu = gpu_q,
            noisy_dev = dev)
        # Cellula per ottenere lo stato nascosto h_t
        self.cell_5 = VQCVariationalLoadingFlexNoisy(
            num_of_input= lstm_internal_size,
            num_of_output= lstm_hidden_size,
            num_of_wires = 4,
            num_of_layers = lstm_cell_num_layers,
            qdevice = qdevice,
            hadamard_gate = True,
            more_entangle = True,
            gpu = gpu_q,
            noisy_dev = dev)
        # Cellula per produrre l'output (che nel caso del decoder equivale alla ricostruzione)
        self.cell_6 = VQCVariationalLoadingFlexNoisy(
            num_of_input= lstm_internal_size,
            num_of_output= lstm_output_size,
            num_of_wires = 4,
            num_of_layers = lstm_cell_num_layers,
            qdevice = qdevice,
            hadamard_gate = True,
            more_entangle = True,
            gpu = gpu_q,
            noisy_dev = dev)

    def get_angles_atan(self, in_x):
        # Trasforma gli input (ad es. tramite arctan) per mappare in un intervallo utile
        return torch.stack([torch.stack([torch.atan(item), torch.atan(item**2)]) for item in in_x])

    def _forward(self, single_item_x, single_item_h, single_item_c):
        # Aggiorna i parametri quantistici per ogni cella
        self.cell_1.var_Q_circuit = self.q_params_1
        self.cell_2.var_Q_circuit = self.q_params_2
        self.cell_3.var_Q_circuit = self.q_params_3
        self.cell_4.var_Q_circuit = self.q_params_4
        self.cell_5.var_Q_circuit = self.q_params_5
        self.cell_6.var_Q_circuit = self.q_params_6

        # Se richiesto, duplica l'input (ad esempio per ripetere lo stesso dato in ingresso)
        single_item_x = torch.cat([single_item_x for _ in range(self.duplicate_time_of_input)])
        # Concatenazione dell'input e dello stato nascosto corrente
        cat = torch.cat([single_item_x, single_item_h])
        res_temp = self.get_angles_atan(cat)

        # Passa i dati attraverso le varie celle quantistiche e applica le attivazioni
        res_from_cell_1 = torch.sigmoid(self.cell_1.forward(res_temp))
        res_from_cell_2 = torch.sigmoid(self.cell_2.forward(res_temp))
        res_from_cell_3 = torch.tanh(self.cell_3.forward(res_temp))
        res_from_cell_4 = torch.sigmoid(self.cell_4.forward(res_temp))

        res_2_mul_3 = torch.mul(res_from_cell_2, res_from_cell_3)
        res_c = torch.mul(single_item_c, res_from_cell_1)
        c_t = res_c + res_2_mul_3

        h_t = self.cell_5.forward(self.get_angles_atan(torch.mul(res_from_cell_4, torch.tanh(c_t))))
        cell_6_res = self.cell_6.forward(self.get_angles_atan(torch.mul(res_from_cell_4, torch.tanh(c_t))))
        out = self.classical_nn_linear(cell_6_res)

        return h_t, c_t, out

    def forward(self, input_sequence_x, initial_h, initial_c, return_all=False):
        """
        Esegue il forward pass sul layer VQLSTM.
        Se return_all è True, restituisce l'intera sequenza di output (utile per il decoder).
        Altrimenti, restituisce solo l'ultimo output (utile per l'encoder).
        """
        h = initial_h.clone().detach()
        c = initial_c.clone().detach()
        outputs = []
        for item in input_sequence_x:
            h, c, out = self._forward(item, h, c)
            outputs.append(out)
        if return_all:
            return h, c, torch.stack(outputs)
        else:
            return h, c, outputs[-1]

# =============================================================================
# Classe AutoencodedVQLSTM – Autoencoder sequenziale
#
# In questa classe:
#   - L'encoder comprime la sequenza di input in uno stato latente.
#   - Il decoder, partendo da uno “start sequence” (ad es. vettori a zero),
#     utilizza lo stato latente per ricostruire (decomprimere) la sequenza.
#
# È possibile scegliere se utilizzare il layer VQLSTM oppure una LSTM classica
# per ciascuno dei due blocchi (encoder e decoder).
# =============================================================================
class AutoencodedVQLSTM(nn.Module):
    def __init__(self, encoder_type, decoder_type,
                 lstm_input_size, lstm_hidden_size, lstm_output_size,
                 lstm_num_qubit, lstm_cell_cat_size, lstm_cell_num_layers,
                 lstm_internal_size, duplicate_time_of_input, as_reservoir,
                 single_y, output_all_h, qdevice, dev, gpu_q):
        super().__init__()
        self.encoder_type = encoder_type
        self.decoder_type = decoder_type
        self.lstm_input_size = lstm_input_size      # Dimensione dell'input originale
        self.lstm_hidden_size = lstm_hidden_size        # Dimensione dello stato nascosto (spazio latente)
        self.lstm_output_size = lstm_output_size      # Dimensione dell'output ricostruito (da solito uguale all'input)

        # ----- Encoder: comprime i dati -----
        if encoder_type == "VQLSTM":
            # L'encoder restituisce lo stato latente finale (senza tutta la sequenza)
            self.encoder = VQLSTM(lstm_input_size=lstm_input_size,
                                  lstm_hidden_size=lstm_hidden_size,
                                  lstm_output_size=lstm_hidden_size,  # compressione in uno spazio di dimensione lstm_hidden_size
                                  lstm_num_qubit=lstm_num_qubit,
                                  lstm_cell_cat_size=lstm_cell_cat_size,
                                  lstm_cell_num_layers=lstm_cell_num_layers,
                                  lstm_internal_size=lstm_internal_size,
                                  duplicate_time_of_input=duplicate_time_of_input,
                                  as_reservoir=as_reservoir,
                                  single_y=single_y,
                                  output_all_h=False,  # solo l'ultimo output
                                  qdevice=qdevice,
                                  dev=dev,
                                  gpu_q=gpu_q)
        elif encoder_type == "LSTM":
            # Per LSTM standard, l'input deve avere shape (seq_len, batch, feature);
            # qui assumiamo batch=1.
            self.encoder = nn.LSTM(input_size=lstm_input_size,
                                   hidden_size=lstm_hidden_size,
                                   num_layers=lstm_cell_num_layers,
                                   batch_first=False)
        else:
            raise ValueError("Tipo di encoder non valido. Scegli 'VQLSTM' o 'LSTM'.")

        # ----- Decoder: decomprime i dati (ricostruzione) -----
        if decoder_type == "VQLSTM":
            # Il decoder prende in ingresso vettori di dimensione lstm_hidden_size (lo stato latente)
            # e ricostruisce sequenze di dimensione lstm_output_size
            self.decoder = VQLSTM(lstm_input_size=lstm_hidden_size,
                                  lstm_hidden_size=lstm_hidden_size,
                                  lstm_output_size=lstm_output_size,
                                  lstm_num_qubit=lstm_num_qubit,
                                  lstm_cell_cat_size=lstm_cell_cat_size,
                                  lstm_cell_num_layers=lstm_cell_num_layers,
                                  lstm_internal_size=lstm_internal_size,
                                  duplicate_time_of_input=duplicate_time_of_input,
                                  as_reservoir=as_reservoir,
                                  single_y=single_y,
                                  output_all_h=True,   # vogliamo l'intera sequenza ricostruita
                                  qdevice=qdevice,
                                  dev=dev,
                                  gpu_q=gpu_q)
        elif decoder_type == "LSTM":
            self.decoder = nn.LSTM(input_size=lstm_hidden_size,
                                   hidden_size=lstm_output_size,
                                   num_layers=lstm_cell_num_layers,
                                   batch_first=False)
        else:
            raise ValueError("Tipo di decoder non valido. Scegli 'VQLSTM' o 'LSTM'.")

    def forward(self, input_seq, h_0, c_0, return_all=False):
        """
        input_seq: sequenza di input di forma (seq_len, feature_dim)
        Il metodo esegue:
          1. L'encoder comprime la sequenza in uno stato latente.
          2. Il decoder, a partire da una sequenza di vettori iniziali (qui: zeri),
             ricostruisce l'intera sequenza.
        """
        seq_len = input_seq.shape[0]

        # ----- Fase di encoding -----
        if self.encoder_type == "VQLSTM":
            # Inizializza gli stati per il VQLSTM
            h0_enc = h_0#torch.zeros(self.lstm_hidden_size, dtype=input_seq.dtype, device=input_seq.device)
            # Per il VQLSTM definiamo anche lo stato della cella interno; lo usiamo così com'è
            c0_enc = c_0#torch.zeros(self.encoder.lstm_internal_size, dtype=input_seq.dtype, device=input_seq.device)
            latent_h, latent_c, _ = self.encoder(input_seq, h0_enc, c0_enc, return_all=False)
        else:  # Encoder LSTM
            # Aggiungiamo batch dimension = 1: (seq_len, 1, input_size)
            input_seq_lstm = input_seq.unsqueeze(1)
            h0_enc = h_0#torch.zeros(self.encoder.num_layers, 1, self.lstm_hidden_size, dtype=input_seq.dtype, device=input_seq.device)
            c0_enc = c_0#torch.zeros(self.encoder.num_layers, 1, self.lstm_hidden_size, dtype=input_seq.dtype, device=input_seq.device)
            encoded_output, (latent_h, latent_c) = self.encoder(input_seq_lstm, (h0_enc, c0_enc))
            # Usiamo lo stato nascosto dell'ultimo layer

            print("latent_h: ", latent_h.shape)
            print("latent_c: ", latent_c.shape)
            latent_h = latent_h[-1, 0, :]  # forma (lstm_hidden_size,)
            latent_c = latent_c[-1, 0, :]

        # ----- Fase di decoding -----
        # Prepara l'input per il decoder: una sequenza di "start tokens" a zero, di lunghezza uguale all'input,
        # con dimensione pari a quella dello stato latente.
        decoder_input = torch.zeros(seq_len, self.lstm_hidden_size, dtype=input_seq.dtype, device=input_seq.device)

        if self.decoder_type == "VQLSTM":
            # Nel decoder VQLSTM, passiamo gli stati latenti come iniziali e chiediamo l'intera sequenza di output
            _, _, reconstruction = self.decoder(decoder_input, latent_h, latent_c, return_all=True)
        else:
            # Per LSTM il decoder si aspetta input shape (seq_len, 1, feature)
            decoder_input_lstm = decoder_input.unsqueeze(1)
            h0_dec = latent_h.unsqueeze(0).repeat(self.decoder.num_layers, 1, self.decoder.hidden_size)  # (num_layers, batch_size, hidden_size)
            c0_dec = latent_c.unsqueeze(0).repeat(self.decoder.num_layers, 1, self.decoder.hidden_size)  # (num_layers, batch_size, hidden_size)
            print("latent_h: ", latent_h.shape)
            print("latent_c: ", latent_c.shape)
            print("h0_dec: ", h0_dec.shape)
            print("c0_dec: ", c0_dec.shape)
            print("decoder_input_lstm: ", decoder_input_lstm.shape)
            reconstruction, _ = self.decoder(decoder_input_lstm, (h0_dec, c0_dec))
            reconstruction = reconstruction.squeeze(1)  # forma (seq_len, lstm_output_size)

        if return_all:
            return reconstruction
        else:
            # Restituisce solo l'ultimo output
            return reconstruction[-1]

# =============================================================================
# Esempio di utilizzo
# =============================================================================
def main2():
    # Impostazioni base
    dtype = torch.DoubleTensor
    device_str = 'cpu'
    
    # Configurazione del dispositivo quantistico
    qdevice = "default.qubit"
    gpu_q = False
    lstm_num_qubit = 4

    # Se si volesse usare un modello di rumore Qiskit:
    use_qiskit_noise_model = False
    dev = None
    if use_qiskit_noise_model:
        # Qui andrebbe definito il noise model appropriato
        noise_model = None  # placeholder
        dev = qml.device('qiskit.aer', wires=lstm_num_qubit, noise_model=noise_model)
    else:
        dev = qml.device("default.qubit", wires=lstm_num_qubit)

    # Parametri del modello
    duplicate_time_of_input = 1
    lstm_input_size = 1       # dimensione dell'input originale
    lstm_hidden_size = 3      # dimensione dello stato latente
    lstm_output_size = 1      # dimensione dell'output ricostruito (da solito uguale all'input)
    lstm_cell_cat_size = lstm_input_size + lstm_hidden_size
    lstm_internal_size = 4
    lstm_cell_num_layers = 2  # numero di layer (sia per encoder che per decoder)
    as_reservoir = False
    single_y = False
    output_all_h = True

    # Creazione dell'autoencoder: scegliendo "VQLSTM" oppure "LSTM" per ciascun blocco
    # In questo esempio utilizziamo VQLSTM per entrambi
    autoencoder = AutoencodedVQLSTM(encoder_type="VQLSTM",
                                    decoder_type="VQLSTM",
                                    lstm_input_size=lstm_input_size,
                                    lstm_hidden_size=lstm_hidden_size,
                                    lstm_output_size=lstm_output_size,
                                    lstm_num_qubit=lstm_num_qubit,
                                    lstm_cell_cat_size=lstm_cell_cat_size,
                                    lstm_cell_num_layers=lstm_cell_num_layers,
                                    lstm_internal_size=lstm_internal_size,
                                    duplicate_time_of_input=duplicate_time_of_input,
                                    as_reservoir=as_reservoir,
                                    single_y=single_y,
                                    output_all_h=output_all_h,
                                    qdevice=qdevice,
                                    dev=dev,
                                    gpu_q=gpu_q).double()

    # Dati di esempio: qui si usa una sequenza sintetica (ad es. una funzione seno o altro)
    seq_len = 10
    # Creiamo una sequenza casuale (o sostituire con i dati reali)
    input_seq = torch.linspace(0, 1, steps=seq_len).reshape(seq_len, 1).double()

    print("Sequenza di input:")
    print(input_seq)

    # Esegui l'autoencoder: la ricostruzione dovrebbe essere la "decompressione" dell'input
    reconstruction = autoencoder(input_seq)
    print("Sequenza ricostruita:")
    print(reconstruction)



def main():
    import matplotlib.pyplot as plt
    import pickle
    from datetime import datetime
    import time
    import os
    import copy

    import torch
    import torch.nn as nn
    import torch.optim as optim

    import pennylane as qml
    from pennylane import numpy as np

    from sklearn.preprocessing import StandardScaler, MinMaxScaler
    import pandas as pd

    # Import custom modules
    #from metaquantum.CircuitComponents import *
    from metaquantum import Optimization
    import qiskit
    import qiskit.providers.aer.noise as noise

    from data.bessel_functions import get_bessel_data  # se necessario, qui non è usato
    from lstm_base_class_AE import AutoencodedVQLSTM
    from lstm_federated_data_prepare import TimeSeriesDataSet

    ## Funzione di costo MSE (utilizzata in pretraining)
    def MSEcost_AE(VQC, X, Y, h_0, c_0, seq_len):
        """Calcola la loss MSE fra le previsioni e il target."""
        loss = nn.MSELoss()
        # Per ogni esempio nel batch, esegue il forward pass del modello
        outputs = torch.stack([VQC.forward(vec.reshape(seq_len, 1), h_0, c_0, return_all=True).reshape(seq_len,) for vec in X])
        output_loss = loss(outputs, Y.reshape(Y.shape[0], seq_len))
        print("LOSS AVG: ", output_loss.item())
        return output_loss

    ## Funzione di costo MSE (utilizzata in forecasting)
    def MSEcost(VQC, X, Y, h_0, c_0, seq_len):
        """Calcola la loss MSE fra le previsioni e il target."""
        loss = nn.MSELoss()
        # Per ogni esempio nel batch, esegue il forward pass del modello
        outputs = torch.stack([VQC.forward(vec.reshape(seq_len, 1), h_0, c_0).reshape(1,) for vec in X])
        output_loss = loss(outputs, Y.reshape(Y.shape[0], 1))
        print("LOSS AVG: ", output_loss.item())
        return output_loss

    def train_epoch_full(opt, VQC, data, h_0, c_0, seq_len, batch_size):
        losses = []
        data_loader = torch.utils.data.DataLoader(data, batch_size=batch_size, shuffle=True)
        for X_train_batch, Y_train_batch in data_loader:
            start_time = time.time()
            opt.zero_grad()
            print("CALCULATING LOSS...")
            loss = MSEcost(VQC=VQC, X=X_train_batch, Y=Y_train_batch, h_0=h_0, c_0=c_0, seq_len=seq_len)
            print("BACKWARD..")
            loss.backward()
            losses.append(loss.detach().cpu().numpy())
            opt.step()
            print("FINISHED OPT. Batch time: ", time.time() - start_time)
            
        losses = np.array(losses)
        return losses.mean()

    def train_epoch_full_AE(opt, VQC, data, h_0, c_0, seq_len, batch_size):
        losses = []
        data_loader = torch.utils.data.DataLoader(data, batch_size=batch_size, shuffle=True)
        for X_train_batch, Y_train_batch in data_loader:
            start_time = time.time()
            opt.zero_grad()
            print("CALCULATING LOSS...")
            loss = MSEcost_AE(VQC=VQC, X=X_train_batch, Y=Y_train_batch, h_0=h_0, c_0=c_0, seq_len=seq_len)
            print("BACKWARD..")
            loss.backward()
            losses.append(loss.detach().cpu().numpy())
            opt.step()
            print("FINISHED OPT. Batch time: ", time.time() - start_time)
            
        losses = np.array(losses)
        return losses.mean()

    def saving(exp_name, exp_index, train_len, iteration_list, train_loss_list, test_loss_list, model, simulation_result, ground_truth):
        file_name = exp_name + "_NO_" + str(exp_index) + "_Epoch_" + str(iteration_list[-1])
        saved_simulation_truth = {
            "simulation_result": simulation_result,
            "ground_truth": ground_truth
        }
        if not os.path.exists(exp_name):
            os.makedirs(exp_name)
        # Salva i dati con pickle
        with open(exp_name + "/" + file_name + "_TRAINING_LOST.txt", "wb") as fp:
            pickle.dump(train_loss_list, fp)
        with open(exp_name + "/" + file_name + "_TESTING_LOST.txt", "wb") as fp:
            pickle.dump(test_loss_list, fp)
        with open(exp_name + "/" + file_name + "_SIMULATION_RESULT.txt", "wb") as fp:
            pickle.dump(saved_simulation_truth, fp)
        torch.save(model.state_dict(), exp_name + "/" + file_name + "_torch_model.pth")
        # Plotting
        plotting_data(exp_name, exp_index, file_name, iteration_list, train_loss_list, test_loss_list)
        plotting_simulation(exp_name, exp_index, file_name, train_len, simulation_result, ground_truth)
        return

    def plotting_data(exp_name, exp_index, file_name, iteration_list, train_loss_list, test_loss_list):
        fig, ax = plt.subplots()
        ax.plot(iteration_list, train_loss_list, '-b', label='Training Loss')
        ax.plot(iteration_list, test_loss_list, '-r', label='Testing Loss')
        ax.legend()
        ax.set(xlabel='Epoch', title=exp_name)
        fig.savefig(exp_name + "/" + file_name + "_" + "loss" + "_" + datetime.now().strftime("NO%Y%m%d%H%M%S") + ".pdf", format='pdf')
        plt.clf()
        return

    def plotting_simulation(exp_name, exp_index, file_name, train_len, simulation_result, ground_truth):
        plt.axvline(x=train_len, c='r', linestyle='--')
        plt.plot(simulation_result, '-')
        plt.plot(ground_truth.detach().numpy(), '--')
        plt.suptitle(exp_name)
        plt.savefig(exp_name + "/" + file_name + "_" + "simulation" + "_" + datetime.now().strftime("NO%Y%m%d%H%M%S") + ".pdf", format='pdf')
        return



    # main
    dtype = torch.DoubleTensor
    device = 'cpu'

    # Impostazione del dispositivo quantistico
    qdevice = "lightning.qubit" 
    gpu_q = False
    lstm_num_qubit = 4  # come esempio

    use_qiskit_noise_model = False
    dev = None
    if use_qiskit_noise_model:
        noise_model = combined_noise_backend_normdist(num_qubits=lstm_num_qubit)
        dev = qml.device('qiskit.aer', wires=lstm_num_qubit, noise_model=noise_model)
    else:
        dev = qml.device("lightning.qubit", wires=lstm_num_qubit)

    # Parametri del modello
    duplicate_time_of_input = 1
    lstm_input_size = 1
    lstm_hidden_size = 3
    lstm_cell_cat_size = lstm_input_size + lstm_hidden_size
    lstm_internal_size = 4
    lstm_output_size = 4  
    lstm_cell_num_layers = 2  # Numero di layer LSTM

    as_reservoir = False

    # Inizializzazione del modello VQLSTM
    model = AutoencodedVQLSTM(encoder_type="LSTM",
                                    decoder_type="LSTM",lstm_input_size=lstm_input_size, 
                    lstm_hidden_size=lstm_hidden_size,
                    lstm_output_size=lstm_output_size,
                    lstm_num_qubit=lstm_num_qubit,
                    lstm_cell_cat_size=lstm_cell_cat_size,
                    lstm_cell_num_layers=lstm_cell_num_layers,
                    lstm_internal_size=lstm_internal_size,
                    duplicate_time_of_input=duplicate_time_of_input,
                    as_reservoir=as_reservoir,
                    single_y=True,
                    output_all_h=False,
                    qdevice=qdevice,
                    dev=dev,
                    gpu_q=gpu_q).double()

    # Caricamento dati: qui si legge un file Excel e si estrae la colonna "Actual Generation [GWh]"
    data = pd.read_excel('data_W_chen.xlsx')
    data = data['Actual Generation [GWh]'].values
    len_train = 0.8
    window_size = 4
    train = data[:int(len_train*len(data))].reshape(-1, 1)
    test = data[int(len_train*len(data))-window_size:].reshape(-1, 1)

    # Normalizzazione in intervallo [-1, 1]
    scaler = MinMaxScaler(feature_range=(-1, 1))
    scaler.fit(train)
    train = scaler.transform(train)
    test = scaler.transform(test)

    prediction_horizon = 1  # Per la fase di forecasting

    # Funzione per suddividere la sequenza (per il forecasting)
    def split_sequence(sequence, n_steps, prediction_horizon):
        X, y = list(), list()
        for i in range(len(sequence)):
            end_ix = i + n_steps
            if end_ix + prediction_horizon - 1 > len(sequence) - 1:
                break
            seq_x = sequence[i:end_ix]
            seq_y = sequence[end_ix + prediction_horizon - 1]
            X.append(seq_x)
            y.append(seq_y)
        return np.array(X), np.array(y)

    x_train, y_train = split_sequence(train, window_size, prediction_horizon)
    x_test, y_test = split_sequence(test, window_size, prediction_horizon)

    x_train = torch.Tensor(x_train).type(dtype)
    y_train = torch.Tensor(y_train).type(dtype)
    x_test = torch.Tensor(x_test).type(dtype)
    y_test = torch.Tensor(y_test).type(dtype)

    # Creazione dei dataset per il training
    train_data = TimeSeriesDataSet(x_train, y_train)
    test_data = TimeSeriesDataSet(x_test, y_test)

    print(x_train.shape, x_test.shape)
    print(y_train.shape, y_test.shape)
    x = torch.cat((x_train, x_test)).type(dtype)
    y = torch.cat((y_train, y_test)).type(dtype)
    print(x.shape, y.shape)

    # Per il pre-training (fase autoencoder), il target sarà l'intera sequenza di input da ricostruire 
    auto_y_train = x_train 
    pretrain_data = TimeSeriesDataSet(x_train, auto_y_train)
        
    h_0 = torch.zeros(lstm_hidden_size,).type(dtype)
    c_0 = torch.zeros(lstm_internal_size,).type(dtype)

    print("First data: ", x_train[0])
    print("Autoencoder target (tutta la finestra): ", auto_y_train)


    # ---------- FASE 1: PRE-TRAINING (Autoencoder) ----------
    num_pretraining_epochs = 100  # ad es. 50 epoche per il pre-training
    pretrain_opt = torch.optim.RMSprop(model.parameters(), lr=0.01, alpha=0.99, eps=1e-08)
    pretrain_loss_list = []
    pretrain_iter_list = []
    print("\n=== Inizio Pre-training (Autoencoder) ===")
    for epoch in range(num_pretraining_epochs):
        print(f'\nPre-training Epoch {epoch+1}')
        # Reinizializza gli stati nascosti per ogni epoca
        if model.encoder_type == "LSTM":
            h_0 = torch.zeros(model.encoder.num_layers, 1, model.lstm_hidden_size).type(dtype)
            c_0 = torch.zeros(model.encoder.num_layers, 1, model.lstm_hidden_size).type(dtype)
        else:
            h_0 = torch.zeros(lstm_hidden_size,).type(dtype)
            c_0 = torch.zeros(lstm_internal_size,).type(dtype)
        loss_epoch = train_epoch_full_AE(opt=pretrain_opt, VQC=model, data=pretrain_data, 
                                        h_0=h_0, c_0=c_0, seq_len=window_size, batch_size=10)
        pretrain_loss_list.append(loss_epoch)
        pretrain_iter_list.append(epoch + 1)

    # Dopo il pre-training, "congela" i pesi dell'encoder in modo che non vengano aggiornati durante la fase di forecasting.
    # Si assume che il modello esponga la parte encoder tramite "model.encoder".

    old_model = model

    # Inizializzazione del modello VQLSTM
    model = AutoencodedVQLSTM(encoder_type="LSTM",
                decoder_type="LSTM",
                lstm_input_size=lstm_input_size, 
                    lstm_hidden_size=lstm_hidden_size,
                    lstm_output_size=lstm_output_size,
                    lstm_num_qubit=lstm_num_qubit,
                    lstm_cell_cat_size=lstm_cell_cat_size,
                    lstm_cell_num_layers=lstm_cell_num_layers,
                    lstm_internal_size=lstm_internal_size,
                    duplicate_time_of_input=duplicate_time_of_input,
                    as_reservoir=as_reservoir,
                    single_y=True,
                    output_all_h=False,
                    qdevice=qdevice,
                    dev=dev,
                    gpu_q=gpu_q).double()

    model.encoder = old_model.encoder


    print("\n=== Congelamento dei pesi dell'encoder ===")
    try:
        for param in model.encoder.parameters():
            param.requires_grad = False
    except AttributeError:
        # Se il modello non espone direttamente "encoder", adattare in base alla struttura interna
        print("ATTENZIONE: il modello non ha l'attributo 'encoder'. Assicurarsi di congelare i pesi del primo layer QLSTM manualmente.")
        # Esempio: per congelare i parametri del primo layer (ipotizzando siano in model.lstm[0])
        # for param in model.lstm[0].parameters():
        #     param.requires_grad = False

    # ---------- FASE 2: TRAINING PER IL FORECASTING ----------
    # Qui usiamo il dataset originale con target y_train (valore futuro)
    exp_name = "VQ_LSTM_AUTOENCODED_TS_MODEL_DATALOADER_WIND_{}_QUBIT".format(lstm_num_qubit)
    if as_reservoir:
        exp_name += "_AS_RESERVOIR"
    if use_qiskit_noise_model:
        exp_name += "_QISKIT_NOISE"
    exp_name += "_{}_QuLAYERS".format(lstm_cell_num_layers)
    exp_index = 3
    train_len = len(x_train)

    # Si può usare lo stesso ottimizzatore (oppure crearne uno nuovo che ottimizzi solo i parametri non congelati)
    forecast_opt = torch.optim.RMSprop(filter(lambda p: p.requires_grad, model.parameters()), lr=0.01, alpha=0.99, eps=1e-08)

    train_loss_for_all_epoch = []
    test_loss_for_all_epoch = []
    iteration_list = []

    num_forecast_epochs = 100  # epoche per il training di forecasting
    print("\n=== Inizio Training per il Forecasting ===")
    for i in range(num_forecast_epochs):
        print(f'\nForecasting Epoch {i+1}')
        iteration_list.append(i + 1)
        h_0 = torch.zeros(lstm_hidden_size,).type(dtype)
        c_0 = torch.zeros(lstm_internal_size,).type(dtype)
        train_loss_epoch = train_epoch_full(opt=forecast_opt, VQC=model, data=train_data, 
                                            h_0=h_0, c_0=c_0, seq_len=window_size, batch_size=10)
        #print("Stati dopo training: h_0 =", h_0, "c_0 =", c_0)
        test_loss = MSEcost(VQC=model, X=x_test, Y=y_test, h_0=h_0, c_0=c_0, seq_len=window_size)
        print("TEST LOSS: ", test_loss.item())
        train_loss_for_all_epoch.append(train_loss_epoch)
        test_loss_for_all_epoch.append(test_loss.detach().numpy())

        # Plot per ogni epoca (opzionale)
        plot_each_epoch = True
        if plot_each_epoch:
            if device == 'cuda':
                total_res = torch.stack([model.forward(vec.reshape(window_size, 1), h_0, c_0).reshape(1,) for vec in x.type(dtype)]).detach().cpu().numpy()
                ground_truth_y = y.clone().detach().cpu()
            else:
                total_res = torch.stack([model.forward(vec.reshape(window_size, 1), h_0, c_0).reshape(1,) for vec in x.type(dtype)]).detach().numpy()
                ground_truth_y = y.clone().detach()

            saving(exp_name=exp_name, exp_index=exp_index, train_len=train_len, iteration_list=iteration_list, 
                    train_loss_list=train_loss_for_all_epoch, test_loss_list=test_loss_for_all_epoch, 
                    model=model, simulation_result=total_res, ground_truth=ground_truth_y)

    # Fine del training
    print("Training completato.")

if __name__ == '__main__':
    main()
