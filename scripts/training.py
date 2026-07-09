from translator.training import training 
from translator.config import settings 


def main(): 
    language_src: str = settings.data_text.language_src 
    language_tgt: str = settings.data_text.language_tgt
    
    seq_len: int = settings.model.seq_len
    batch_size: int = settings.model.training.batch_size 
    d_model: int = settings.model.d_model 
    n_epochs: int = settings.model.training.epochs 
    dd_df: int = settings.model.dd_df 
    n_head: int= settings.model.n_head 
    n_encoder_layer: int = settings.model.n_encoder_layer
    n_decoder_layer: int = settings.model.n_decoder_layer 
    dropout: float = settings.model.training.dropout
    save_weight_name: str = settings.model.save_model_name
    load_prev_weight: bool = settings.model.training.load_prev_weight
    
    
    training(language_src=language_src, 
             language_tgt=language_tgt, 
             seq_len=seq_len,
             batch_size=batch_size, 
             d_model= d_model,
             n_epochs=n_epochs,
             dd_df=dd_df,
             n_head = n_head, 
             n_encoder_layer=n_encoder_layer, 
             n_decoder_layer=n_decoder_layer, 
             dropout= dropout, 
             save_weight_name=save_weight_name, 
             load_prev_weight = load_prev_weight
             )
    
if __name__=="__main__":
    main()