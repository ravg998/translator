from translator.eval import eval 
from translator.config import settings 


def main(): 
    language_src: str = settings.data_text.language_src
    language_tgt: str = settings.data_text.language_tgt
    
    sentence_to_translate: str = input(f"Sentence to be translated from {language_src.upper()} to {language_tgt.upper()}:\n")
    
    d_model: int = settings.model.d_model
    dd_df: int = settings.model.dd_df 
    n_head: int = settings.model.n_head 
    seq_len: int = settings.model.seq_len 
    
    n_encoder_layer: int = settings.model.n_encoder_layer
    n_decoder_layer: int = settings.model.n_decoder_layer   

    translation: str = eval(language_src=language_src, 
         language_tgt=language_tgt, 
         d_model=d_model, 
         dd_df = dd_df, 
         n_head=n_head,
         seq_len = seq_len, 
         n_encoder_layer=n_encoder_layer, 
         n_decoder_layer=n_decoder_layer, 
         sentence_to_translate=sentence_to_translate
         )
    
    print(f"TRANSLATION:\n{translation}")
        
if __name__=="__main__":
    main()