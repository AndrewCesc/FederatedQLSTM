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
                                   batch_first=True)
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
                                   batch_first=True)
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
            print(input_seq_lstm.shape)
            print(h0_enc.shape)
            print(c0_enc.shape)
            encoded_output, (latent_h, latent_c) = self.encoder(input_seq_lstm, (h0_enc, c0_enc))
            # Usiamo lo stato nascosto dell'ultimo layer
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
            # Per LSTM il decoder si aspetta input shape (seq_len, batch, feature)
            decoder_input_lstm = decoder_input.unsqueeze(1)
            h0_dec = latent_h.unsqueeze(0)  # forma (1, 1, lstm_hidden_size)
            c0_dec = latent_c.unsqueeze(0)
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
def main():
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

if __name__ == '__main__':
    main()
