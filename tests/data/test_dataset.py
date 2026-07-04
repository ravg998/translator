import pytest 
from translator.data import TokenDataset
from tokenizers import Tokenizer 
from tokenizers.models import WordLevel 
from tokenizers.pre_tokenizers import Whitespace 
from tokenizers.trainers import WordLevelTrainer


parameters: list[dict[str, list[dict[str, str]]]] = [
    {"translation": 
        [{"en": "Hello World",
         "fr": "Bonjour le monde!"}
         ]
        },
    {"translation": 
        [{"en": "Hello how are you doing",
         "fr": "Bonjour comment allez-vous"}
         ]
        }
]
SEQ_LEN: int = 352

@pytest.fixture()
def tokenizer_test() -> Tokenizer: 
    special_token: list[str] = ["[SOS]", "[EOS]", "[PAD]", "[UNK]"]
    tokenizer = Tokenizer(WordLevel(unk_token = "[UNK]"))
    tokenizer.pre_tokenizer = Whitespace()
    trainer = WordLevelTrainer(special_tokens=special_token, 
                               min_frequency=1)
    
    corpus: list[str] = ["Hello World", "Bonjour comment allez-vous", "Il fait très beau aujourd'hui"]
    tokenizer.train_from_iterator(corpus, trainer)
    
    return tokenizer

@pytest.fixture(params = parameters)
def token_dataset(request: pytest.fixture, tokenizer_test: Tokenizer) -> TokenDataset: 
    data_set = request.param
    
    return TokenDataset(dataset=data_set, 
                        token_src = tokenizer_test, 
                        token_tgt = tokenizer_test, 
                        pad_token="[PAD]", 
                        sos_token = "[SOS]", 
                        eos_token= "[EOS]",
                        language_src="en", 
                        language_tgt="fr",
                        seq_len = SEQ_LEN)

class TestTokenDataset: 
    def test_encoder_input_shape(self, token_dataset: TokenDataset) -> None: 
        encoder_input = token_dataset[0]["encoder_input"]
        
        assert encoder_input.shape == (SEQ_LEN,)
        
    def test_encoder_mask_shape(self, token_dataset: TokenDataset) -> None: 
        encoder_mask = token_dataset[0]["encoder_mask"]
        assert encoder_mask.shape == (1, 1, SEQ_LEN)
        
    def test_decoder_tgt_shape(self, token_dataset: TokenDataset) -> None: 
        decoder_tgt = token_dataset[0]["decoder_tgt"]
        assert decoder_tgt.shape == (SEQ_LEN,)
    
    def test_label_shape(self, token_dataset: TokenDataset) -> None: 
        label = token_dataset[0]["label"]
        assert label.shape == (SEQ_LEN,)
    
    def test_decoder_mask_shape(self, token_dataset: TokenDataset) -> None: 
        decoder_mask = token_dataset[0]["decoder_mask"]
        assert decoder_mask.shape == (1, SEQ_LEN, SEQ_LEN)

        
