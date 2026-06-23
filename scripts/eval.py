from translator.eval import eval 
from translator.config import settings 


def main(): 
    language_src: str = settings.data_text.language_src
    language_tgt: str = settings.data_text.language_tgt
    
    d_model: int = settings.model.d_model
    dd_df: int = settings.model.dd_df 
    n_head: int = settings.model.n_head 
    seq_len: int = settings.model.seq_len 
    
    n_encoder_layer: int = settings.model.n_encoder_layer
    n_decoder_layer: int = settings.model.n_decoder_layer   
    dict_sentence = {'en': "I still say 'our home,' because we left out house 50 years ago although the house no longer belongs to us.", 'fr': 'Je continue à dire « chez nous », bien que la maison ne nous appartienne plus.'}
    sentence_to_translate: str = dict_sentence["en"]
    
    eval(language_src=language_src, 
         language_tgt=language_tgt, 
         d_model=d_model, 
         dd_df = dd_df, 
         n_head=n_head,
         seq_len = seq_len, 
         n_encoder_layer=n_encoder_layer, 
         n_decoder_layer=n_decoder_layer, 
         sentence_to_translate=sentence_to_translate
         )
        
        
if __name__=="__main__":
    main()