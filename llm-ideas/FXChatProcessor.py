from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.llms import OpenAI
from langchain.prompts.example_selector import SemanticSimilarityExampleSelector
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.prompts import PromptTemplate

import os
import openai

os.environ["OPENAI_API_KEY"] = "your key here"
openai.api_key = "your key here"

from numpy import dot
from numpy.linalg import norm

def cosine_similarity(list_1, list_2):
    cos_sim = dot(list_1, list_2) / (norm(list_1) * norm(list_2))
    return cos_sim

template = """
You are an agent that will read a message from a fx trader and you will decipher the message to a json file so that you can feed it to an api. The JSON consist of the following terms

Entity -
Text, only allow EB or IB, EB refers to the European Bank or known as SANV, SA/NV, SA, IB is the institional bank and should be the default if it is not clear
Counterparty - 
This must be a 4 letter code referring to what the trade is with, for example DEUT, NFXL
Notional - 
This should be the notional amount of the trade entered as a number, note that conventionally if the number is small and less than 1000 then the conventions are the following
25 means 25000000
2.5 means 2500000
45.3mio or 45.3m means  45300000
2.56 yds or yard means 2560000000
a yard means 1000000000
This must be present otherwise the trade is invalid and the output should say so
Notional_Currency - This is the currency of the notional of the trade
Currency1 - The first 3 letter currency imentioned in a pair, if the currency is not specified, assume it is USD
Currency2 - The second 3 letter currency mentioned in a pair, if the currency is not specified, assume it is USD
Trade Type - Text field, it can be swap, ndf, if nothing mentioned, it is likely to be swap, an outright trade is a swap with first leg as o/n
Direction - It is either BS or SB based the the currency pair, if nothing is mentioned, it is TwoWay
Tenor - Length of the trade can be overnight which will be "o/n" can be given as "on", tomorrow night as "tn" or "t/n", then standard tenors like 30D, 1M ..etc. you may have to translate this tenor
If you are not able to populate all the fields, tell the user the trade is incomplete and why
if you have something like sp-1m, this means tenor is "on" and TenorEnd is "1M"
TenorEnd - Should bet the end date of swap, use the samec conventions as Tenor

{examples_list}

{format_instructions}

Input: {input}
Output:
"""

model_name = 'text-davinci-003'
temperature = 0.0
model = OpenAI(model_name=model_name, temperature=temperature)


example_prompt = PromptTemplate(
    input_variables=["input", "output"],
    template="Input: {input}\nOutput: {output}",
)

examples = [
    {"input": "second trade: can i please trade $25 USDCNH 6month swap with NFXL", 
     "output": """{ "Entity" : "IB", "Counterparty" : "NFXL", "Notional" : 25000000, "Notional_currency" : "USD", "Currency1" : "USD", "Currency2" : "CNH", "Trade_type" : "swap", "Direction" : "TwoWay", "Tenor" : "ON", "TenorEnd" : "6M" }"""},
    {"input": "Can IB b/s 300mio dkk vs usd t/n with NFXL asking for apporval", 
     "output": """{ "Entity" : "IB", "Counterparty" : "NFXL", "Notional" : 300000000, "Notional_currency" : "DKK", "Currency1" : "DKK", "Currency2" : "USD", "Trade_type" : "swap", "Direction" : "BS", "Tenor" : "ON", "TenorEnd" : "TN" }"""},
    {"input": "can i please trade about $4 usdthb 3M swap with DEUL on sa", 
     "output": """{ "Entity" : "EB", "Counterparty" : "DEUL", "Notional" : 4000000, "Notional_currency" : "USD", "Currency1" : "USD", "Currency2" : "THB", "Trade_type" : "swap", "Direction" : "TwoWay", "Tenor" : "ON", "TenorEnd" : "3M" }"""},
    {"input": "b/s thb vs dkk ndf 1m swap jpm", 
     "output": "error - no notional defined"},
    {"input": "Can sanv do ils sp-1m with hsbc?",
     "output": "error - no notional defined"},
    {"input": "Can sanv do 24.55mio ils sp-1m with hsbc?",
     "output": """{ "Entity" : "EB", "Counterparty" : "HSBC", "Notional" : 24550000, "Notional_currency" : "ILS", "Currency1" : "ILS", "Currency2" : "USD", "Trade_type" : "swap", "Direction" : "TwoWay", "Tenor" : "ON", "TenorEnd" : "1M" }"""},
    {"input": "b/s 2yds thb vs dkk ndf 1m ndf with jpm",
     "output": """{ "Entity" : "IB", "Counterparty" : "JPM", "Notional" : 2000000000, "Notional_currency" : "THB", "Currency1" : "THB", "Currency2" : "DKK", "Trade_type" : "ndf", "Direction" : "TwoWay", "Tenor" : "ON", "TenorEnd" : "1M" }"""},
    {"input": "another ccar ask, $2 usdinr outright 1m ag sgel",
     "output": """{ "Entity" : "IB", "Counterparty" : "SGEL", "Notional" : 2000000, "Notional_currency" : "USD", "Currency1" : "USD", "Currency2" : "INR", "Trade_type" : "swap", "Direction" : "TwoWay", "Tenor" : "ON", "TenorEnd" : "1M" }"""},
    {"input": "can i trade 100 bucks versus euro with hsbc outright one month with sa",
     "output": """{ "Entity" : "EB", "Counterparty" : "HSBC", "Notional" : 100000000, "Notional_currency" : "USD", "Currency1" : "USD", "Currency2" : "EUR", "Trade_type" : "swap", "Direction" : "TwoWay", "Tenor" : "ON", "TenorEnd" : "1M" }"""},
    {"input": "can i trade 2 yards yen versus euro with hsbc outright three months with sa",
     "output": """{ "Entity" : "EB", "Counterparty" : "HSBC", "Notional" : 2000000000, "Notional_currency" : "JPY", "Currency1" : "JPY", "Currency2" : "EUR", "Trade_type" : "swap", "Direction" : "TwoWay", "Tenor" : "ON", "TenorEnd" : "3M" }"""},
]

example_selector = SemanticSimilarityExampleSelector.from_examples(examples, OpenAIEmbeddings(), Chroma, k=5)

from pydantic import BaseModel, Field, validator
from langchain.output_parsers import PydanticOutputParser, OutputFixingParser
from typing import Union, Optional
from enum import Enum

# Define your desired data structure.
class Tenor(str, Enum):
    ON = "ON"
    TN = "TN"
    
class TradeData(BaseModel):
    Entity: str
    Counterparty: str
    Notional: int
    Notional_Currency: str
    Currency1: str
    Currency2: str
    Trade_Type: str
    Direction: str
    Tenor: Union[int, str, Tenor]
    TenorEnd: Union[int, str, Tenor]
    
    @validator('Tenor', 'TenorEnd')
    def validate_tenor(cls, v):
        if isinstance(v, int):
            return v
        elif v.upper() == "ON":
            return Tenor.ON
        elif v.upper() == "TN":
            return Tenor.TN
        else:
            try:
                if v[-1].lower() == 'd':
                    return int(v[:-1])
                elif v[-1].lower() == 'm':
                    return int(v[:-1]) * 30
                elif v[-1].lower() == 'y':
                    return int(v[:-1]) * 365
                else:
                    raise ValueError
            except:
                raise ValueError('Invalid tenor value')
    
class Trade(BaseModel):
    input: str = Field(description="the chat input from the chat")
    trade_json: Optional[TradeData] = Field(None, description="trade represented as json")
    error: Optional[str] = Field(None, description="error message if there are any")

# Set up a parser + inject instructions into the prompt template.
parser = PydanticOutputParser(pydantic_object=Trade)

def chat_parser(chat_input, verbose=True):
    if chat_input == '': return ''
    examples_objects = example_selector.select_examples({"input": chat_input})
    parser = PydanticOutputParser(pydantic_object=Trade)
    examples = '\n'.join(['input:\n' + r['input'] + "\noutput:\n" + r['output'] +'\n' for r in examples_objects])
    prompt = PromptTemplate(input_variables=["input"], template=template, partial_variables={"format_instructions": parser.get_format_instructions(), "examples_list" : examples})
    _input = prompt.format_prompt(input=chat_input)
    output = model(_input.to_string())
    if verbose: print(output)
    return output
    