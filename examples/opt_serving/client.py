import argparse
from typing import Dict, Optional, Union, Sequence

import requests

DEFAULT_URL = "https://opt.alpa.ai/"


class Client(object):

    def __init__(self, url: Optional[str] = None) -> None:
        if url is None:
            url = DEFAULT_URL

        self.completions_url = url + "/completions"
        self.logprobs_url = url + "/completions"

    def completions(
        self,
        prompt: Union[str, Sequence[str], Sequence[int], Sequence[Sequence[int]]],
        min_tokens: int = 0,
        max_tokens: int = 32,
        top_p: float = 1.0,
        temperature: float = 1.0,
        echo: bool = True,
    ) -> Dict:
        """
        Generation API.
        Parameters match those of the OpenAI API.
        https://beta.openai.com/docs/api-reference/completions/create

        Args:
          prompt: a list of tokenized inputs.
          min_tokens: The minimum number of tokens to generate.
          max_tokens: The maximum number of tokens to generate.
          temperature: What sampling temperature to use.
          top_p: The nucleus sampling probability.
          echo: if true, returned text/tokens/scores includes the prompt.
        """
        pload = {
            "prompt": prompt,
            "min_tokens": min_tokens,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "echo": echo,
        }
        result = requests.post(self.completions_url, json=pload)
        return result.json()

    def logprobs(
        self,
        prompt: Union[str, Sequence[str], Sequence[int], Sequence[Sequence[int]]],
        top_p: float = 1.0,
        top_k: int = 100000,
        cache_id: Optional = None) -> Dict:
        """TODO(yangkevin2)"""
        pload = {
            "prompt": prompt,
            "top_k": top_k,
            "redirect_logprobs": True
        }
        if cache_id:
            pload["cache_id"] = cache_id
        result = requests.post(self.logprobs_url, json=pload)
        return result.json()
