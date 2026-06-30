import translator as translator 
from  translator import settings

def main(): 
    data = translator.DataSource(language_src = settings.data_text.language_src,
                                 language_tgt = settings.data_text.language_tgt, 
                                 data_source = settings.data_text.data_source, 
                                 save_path = settings.data_path.data_source)
    
    print(data.get_data()[:10])
    
if __name__=="__main__": 
    main()