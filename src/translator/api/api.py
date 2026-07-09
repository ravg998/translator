from flask import Flask, request
import translator.eval as eval 

app = Flask(__name__)

@app.get("/translate")
def translate_sentence() -> str: 
    model=request.args.get("model")
    sentence_to_translate = request.args.get("sentence")
    print(f"{sentence_to_translate=}")
    return eval.eval(model, sentence_to_translate)

if __name__=="__main__": 
    app.run(debug=True)