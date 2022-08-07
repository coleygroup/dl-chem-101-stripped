
import csv
import os
from os import path
from typing import Union, Tuple
import rdkit
import torch
from smiles_lstm.model.smiles_dataset import Dataset
from smiles_lstm.model.smiles_vocabulary import SMILESTokenizer
from smiles_lstm.model.smiles_lstm import SmilesLSTM
from smiles_lstm.utils.misc import draw_smiles, progress_bar, save_smiles
import smiles_lstm.utils.load as load

rdkit.rdBase.DisableLog("rdApp.error")


class SmilesTrainer():
    
    def __init__(self, model: SmilesLSTM, input_smiles : Union[dict, str],
                 epochs : int=10, learning_rate : float=0.0001,
                 batch_size : int=250, shuffle : bool=True,
                 augment : int=0, output_model_path : str="./output/", start_epoch : int=0,
                 learning_rate_scheduler : str="StepLR", gamma : float=0.8,
                 eval_num_samples : int=64, eval_batch_size : int=64) -> None:
        
        
        self._model = model

        
        self._batch_size        = batch_size
        self._learning_rate     = learning_rate
        self._epochs            = epochs
        self._start_epoch       = start_epoch
        self._output_model_path = output_model_path
        self._shuffle           = shuffle
        self._use_augmentation  = augment
        self._eval_num_samples  = eval_num_samples
        self._eval_batch_size   = eval_batch_size

        
        (self._train_dataloader,
         self._test_dataloader,
         self._valid_dataloader) = self._load_smiles(input_smiles=input_smiles)

        
        self._optimizer = torch.optim.Adam(params=self._model.network.parameters(),
                                           lr=self._learning_rate)

        if learning_rate_scheduler == "CosineAnnealingLR":
            
            self._scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer=self._optimizer,
                T_max=self._epochs,
            )
        elif learning_rate_scheduler == "StepLR":
            
            self._scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer=self._optimizer,
                step_size=1,
                gamma=gamma,
            )
        else:
            raise ValueError("Please enter a valid value for ´learning_rate_scheduler´.")

        
        if not os.path.exists(self._output_model_path):
            os.makedirs(self._output_model_path)
      
        
        params_filename                = f"{self._output_model_path}SmilesTrainer_params.csv"
        self._training_output_filename = f"{self._output_model_path}SmilesTrainer_training.csv"

        
        with open(params_filename, "w", encoding="utf-8") as params_file:
            params_writer = csv.writer(params_file)
            param_names  = ["model type", "batch size", "learning rate",
                            "epochs", "start_epoch", "use shuffle",
                            "use augmentation", "eval num samples",
                            "eval batch size", "learning rate scheduler"]
            param_values = ["SmilesTrainer", self._batch_size, self._learning_rate,
                            self._epochs, self._start_epoch, self._shuffle,
                            self._use_augmentation, self._eval_num_samples,
                            self._eval_batch_size, learning_rate_scheduler]
            params_writer.writerow(param_names)  
            params_writer.writerow(param_values)

        
        self._train_loss      = None
        self._valid_loss      = None
        self._best_valid_loss = None
        self._best_epoch      = None

        
        with open(self._training_output_filename, "w", encoding="utf-8") as training_file:
            training_writer = csv.writer(training_file)
            header = ["epoch", "learning rate", "training loss",
                      "validation loss", "fraction valid"]
            training_writer.writerow(header)

    def run(self):
        
        
        for epoch in range(self._start_epoch, self._epochs):

            self._train_epoch(self._train_dataloader)
            self._valid_epoch(self._valid_dataloader)

            
            sampled_smiles, nlls = self._model.sample_smiles(num=self._eval_num_samples,
                                                             batch_size=self._eval_batch_size)
            fraction_valid       = draw_smiles(
                path=f"{self._output_model_path}sampled_epoch{epoch}.png",
                smiles_list=sampled_smiles
            )
            save_smiles(smiles=sampled_smiles,
                        output_filename=f"{self._output_model_path}sampled_step{epoch}.smi")

            
            learning_rate = self._optimizer.param_groups[0]["lr"]
            with open(self._training_output_filename, "a", encoding="utf-8") as training_file:
                training_writer = csv.writer(training_file)
                progress        = [epoch,
                                   learning_rate,
                                   self._train_loss.item(),
                                   self._valid_loss.item(),
                                   fraction_valid]
                training_writer.writerow(progress)

            self._save_checkpoint(epoch)

    def _save_checkpoint(self, epoch : int) -> None:
        
        
        if self._best_valid_loss is None:
            self._best_valid_loss = self._valid_loss
        elif self._valid_loss < self._best_valid_loss:
            self._best_valid_loss = self._valid_loss
            self._best_epoch = epoch

        self._save_current_model(epoch)

    def _train_epoch(self, train_dataloader : torch.utils.data.DataLoader):
        
        loss_tensor = torch.zeros(len(train_dataloader))
        self._model.network.train()
        dataloader_progress_bar = progress_bar(iterable=train_dataloader,
                                               total=len(train_dataloader))
        for batch_idx, batch in enumerate(dataloader_progress_bar):
            input_vectors          = batch.long()
            loss                   = self._calculate_loss(input_vectors)
            loss_tensor[batch_idx] = loss

            self._model.network.zero_grad()  
            self._optimizer.zero_grad()      
            loss.backward()
            self._optimizer.step()

        
        self._train_loss = torch.mean(loss_tensor)

        
        self._scheduler.step()

    def _valid_epoch(self, valid_dataloader : torch.utils.data.DataLoader):
        
        loss_tensor = torch.zeros(len(valid_dataloader))
        self._model.network.eval()
        dataloader_progress_bar = progress_bar(iterable=valid_dataloader,
                                               total=len(valid_dataloader))
        with torch.no_grad():
            for batch_idx, batch in enumerate(dataloader_progress_bar):
                input_vectors          = batch.long()
                loss                   = self._calculate_loss(input_vectors)
                loss_tensor[batch_idx] = loss

        
        self._valid_loss = torch.mean(loss_tensor)

    def _initialize_dataloader(self, smiles_list : list) -> torch.utils.data.DataLoader:
        
        if self._use_augmentation:
            smiles_list_augmented = []
            for smiles in smiles_list:
                smiles_list_augmented += self._augment(smiles=smiles,
                                                       n_permutations=self._use_augmentation)
        smiles_list += smiles_list_augmented

        dataset = Dataset(smiles_list=smiles_list,
                          vocabulary=self._model.vocabulary,
                          tokenizer=SMILESTokenizer())
        if len(dataset) == 0:
            raise IOError(f"No valid entries are present in the "
                          f"supplied file: {path}")

        dataloader = torch.utils.data.DataLoader(dataset,
                                                 batch_size=self._batch_size,
                                                 shuffle=self._shuffle,
                                                 collate_fn=Dataset.collate_fn,
                                                 drop_last=True)
        return dataloader

    def _augment(self, smiles : str, n_permutations : int) -> list:
        
        molecule = rdkit.Chem.MolFromSmiles(smiles)

        try:
            permutations = [rdkit.Chem.MolToSmiles(molecule,
                                                   canonical=False,
                                                   doRandom=True,
                                                   isomericSmiles=False)
                            for _ in range(n_permutations)]
        except RuntimeError:
            permutations = [smiles]

        return permutations


    def _calculate_loss(self, input_vectors : torch.Tensor) -> torch.Tensor:
        
        log_p = self._model.likelihood(input_vectors)
        return log_p.mean()

    def _save_current_model(self, epoch : int) -> None:
        
        model_path = f"{self._output_model_path}model.{epoch}.pth"
        self._model.save_state(path=model_path)

    def _load_smiles(self, input_smiles : Union[dict, str]) ->                     Tuple[list, list, list]:
        
        
        if isinstance(input_smiles, str):
            
            train_smiles = load.smiles(path=f"{input_smiles}train.smi")
            test_smiles  = load.smiles(path=f"{input_smiles}test.smi")
            valid_smiles = load.smiles(path=f"{input_smiles}valid.smi")
        elif isinstance(input_smiles, dict):
            
            train_smiles = input_smiles["train"]
            test_smiles  = input_smiles["test"]
            valid_smiles = input_smiles["valid"]
        else:
            raise NotImplementedError

        
        train_dataloader = self._initialize_dataloader(smiles_list=train_smiles)
        test_dataloader  = self._initialize_dataloader(smiles_list=test_smiles)
        valid_dataloader = self._initialize_dataloader(smiles_list=valid_smiles)

        return train_dataloader, test_dataloader, valid_dataloader
