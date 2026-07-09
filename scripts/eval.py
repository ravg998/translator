from translator.eval import eval 
from translator.config import settings 


def main(): 

    
    sentence_to_translate: str = input(f"Sentence to be translated:\n")
    
    weight_file_name: str = settings.model.save_model_name
    translation: str = eval(weight_file = weight_file_name, 
         sentence_to_translate=sentence_to_translate
         )
    
    print(f"TRANSLATION:\n{translation}")
        
if __name__=="__main__":
    main()