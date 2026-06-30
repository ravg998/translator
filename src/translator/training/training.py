import torch 
import torch.nn as nn 
from torch.utils.data import DataLoader
from translator.data import DataSource, TokenLoad, TokenDataset
import translator.model as translator_model 
from translator.utils import setup_device
from tokenizers import Tokenizer
from translator.config import settings 
from tqdm import tqdm
import logging 

logger = logging.getLogger(__name__)
def training(language_src: str, 
             language_tgt: str, 
             seq_len: str, 
             batch_size: str, 
             d_model: int, 
             n_epochs: int,
             dd_df: int, 
             n_head: int, 
             n_encoder_layer: int,
             n_decoder_layer: int
             ) -> None: 
    # DATA
    data_source: dict = DataSource(language_src=language_src, 
                                   language_tgt=language_tgt, 
                                   data_source = settings.data_text.data_source, 
                                   save_path = settings.data_path.data_source
                                   ).get_data()[:1_0]
    tqdm.write(str(data_source))

    token_src: Tokenizer = TokenLoad(data=data_source, 
                                     language=language_src).get_token(force_load = True)
    
    token_tgt: Tokenizer = TokenLoad(data=data_source, 
                                     language=language_tgt).get_token(force_load = True)
    

    data_set: TokenDataset = TokenDataset(dataset = data_source, 
                                          token_src = token_src, 
                                          token_tgt = token_tgt, 
                                          seq_len = seq_len, 
                                          language_src=language_src, 
                                          language_tgt=language_tgt, 
                                          pad_token = settings.tokenizer_cfg.padding, 
                                          sos_token = settings.tokenizer_cfg.sos, 
                                          eos_token = settings.tokenizer_cfg.eos
                                          )
    
    data_loader = DataLoader(dataset=data_set, 
                             batch_size=batch_size, 
                             shuffle=True)
    
    # MODEL
    device = setup_device()
    model = translator_model.Transformer(d_model=d_model, 
                                         n_head=  n_head, 
                                         dd_df = dd_df, 
                                         seq_len=seq_len,
                                         n_encoder_layer=n_encoder_layer, 
                                         n_decoder_layer=n_decoder_layer,
                                         vocab_size_src=token_src.get_vocab_size(),
                                         vocab_size_tgt=token_tgt.get_vocab_size()
                                         )
    model.to(device)
    criterion = nn.CrossEntropyLoss(ignore_index = token_tgt.token_to_id(settings.tokenizer_cfg.padding))
    optimizer: torch.optim.Adam = torch.optim.Adam(model.parameters(), lr=settings.model.training.lr)
    
    for epoch in tqdm(range(n_epochs), position =0, desc="Epoch"):
        for batch in tqdm(data_loader, position=1, desc = "Batch"):
            optimizer.zero_grad()
            
            encoder_input = batch["encoder_input"].to(device)
            encoder_mask = batch["encoder_mask"].to(device)
            decoder_tgt = batch["decoder_tgt"].to(device)
            label = batch["label"].long().to(device)
            decoder_mask = batch["decoder_mask"].to(device)
            
            output = model(x_src= encoder_input, 
                        x_tgt = decoder_tgt, 
                        mask_pad = encoder_mask, 
                        mask_hid = decoder_mask)
        
            loss = criterion(output.view(-1,output.size(-1)
                            ), label.view(-1))
            loss.backward()
            optimizer.step()
            
        tqdm.write(f"Loss: {loss.item(): ,.5f}")
    
            
        logger.info(f"Epoch: {epoch +1:,.0f}/{n_epochs:,.0f} ({epoch/n_epochs:,.0%}) | Loss: {loss.item():,.5f}")
    
    settings.data_path.weight.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), 
               settings.data_path.weight / f'{language_src}_to_{language_tgt}.pt')
    
if __name__=="__main__": 
    training()