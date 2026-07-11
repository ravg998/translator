from flask import Flask, request
import translator.eval as eval 

app = Flask(__name__)

@app.get("/translate")
def translate_sentence() -> str: 
    model=request.args.get("model")
    sentence_to_translate = request.args.get("sentence").replace("%", " ")
    sentence_translated: str =  eval.eval(model, sentence_to_translate)
    
    return f"Original sentence: {sentence_to_translate}\n ---->   Translation:{sentence_translated}"

if __name__=="__main__": 
    app.run(debug=True)