import pytest 
from translator.training import filter_sentence_too_long


parameters: list[dict] = [{"translation": [{"en": "Hello World", 
                                            "fr": "Bonjour le monde"
                                            }, 
                                           {"en": "I am writing a very long sentence", 
                                            "fr": "J'écris une phrase très longue"
                                            }
                                           ],
                           "max_seq_len": 3, 
                           "language_src": "en", 
                           "language_tgt": "fr", 
                           "output": {"translation": [{"en": "Hello World", 
                                            "fr": "Bonjour le monde"
                                            }
                           ]
                           }
                           } 
                          
                          ]


@pytest.fixture(params=parameters)
def data_source(request: pytest.fixture) -> dict: 
    return {"translation": request.param["translation"]}

@pytest.fixture(params=parameters)
def arguments(request: pytest.fixture) -> dict:
    arguments = request.param.copy()
    arguments.pop("translation")
    
    return arguments

class TestSeqLengthFilter: 
    def test_seq_length_filter(self, 
                               data_source: dict, 
                               arguments: dict
                               ): 
        print(arguments)
        output = arguments.pop("output")
        assert TestSeqLengthFilter._evaluate_dict(filter_sentence_too_long(data_source,
                                                                           **arguments),
                                                  output)
    @staticmethod
    def _evaluate_dict(dict_1, dict_2) -> bool: 
        if len(dict_1) != len(dict_2):
            return False
        print(dict_1)
        print(dict_2)
        for t1, t2 in zip(dict_1["translation"], 
                          dict_2["translation"]): 
            if t1.keys() != t2.keys() or list(t1.values()) != list(t2.values()):
                return False
        return True