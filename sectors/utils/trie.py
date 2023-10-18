import numpy as np
from copy import deepcopy
from typing import Dict, List


class Trie(object):
    def __init__(
        self,
        bos_token_id,
        sep_token_id,
        eos_token_id,
        col_token_id=None,
        pad_token_id=None,
        sequences: List[List[int]] = [],
    ):
        self.trie_dict = {}
        self.len = 0
        self.bos_token_id = bos_token_id
        self.sep_token_id = sep_token_id
        self.eos_token_id = eos_token_id
        self.col_token_id = col_token_id
        self.pad_token_id = pad_token_id
        if sequences:
            for sequence in sequences:
                if sequence[-1] == self.eos_token_id:
                    sequence = sequence[:-1]
                if sequence[0] == self.bos_token_id:
                    sequence = sequence[1:]
                Trie._add_to_trie(sequence + [self.sep_token_id], self.trie_dict)
                Trie._add_to_trie(sequence + [self.eos_token_id], self.trie_dict)
                self.len += 1
        self.n_previous_labels = {}
        self.reduced_trie = {}
        self.lastnex = {}
        self.last_sequence = {}

    def add(self, sequence: List[int]):
        Trie._add_to_trie(sequence, self.trie_dict)
        self.len += 1

    def get(self, batch_id: int, prefix_sequence: List[int]) -> List[int]:
        self.last_sequence[batch_id] = prefix_sequence
        if prefix_sequence[0] == self.bos_token_id:
            prefix_sequence = prefix_sequence[1:]
        if self.col_token_id:
            colon_indices = np.where(np.array(prefix_sequence) == self.col_token_id)[0]
            last_colon_index = colon_indices[-1]
            prefix_sequence = prefix_sequence[last_colon_index + 1 :]
        if not prefix_sequence:
            self.n_previous_labels[batch_id] = 0
        if self.sep_token_id in prefix_sequence:
            sep_indices = np.where(np.array(prefix_sequence) == self.sep_token_id)[0]
            last_sep_index = sep_indices[-1]
            current_sequence = prefix_sequence[last_sep_index + 1 :]
            if current_sequence == []:
                self.n_previous_labels[batch_id] += 1
                if self.n_previous_labels[batch_id] == 1:
                    previous_seq = prefix_sequence[:last_sep_index]
                    self.reduced_trie[batch_id] = self.delete(
                        previous_seq, self.trie_dict
                    )
                elif self.n_previous_labels[batch_id] > 1:
                    previous_seq = prefix_sequence[sep_indices[-2] + 1 : last_sep_index]
                    self.reduced_trie[batch_id] = self.delete(
                        previous_seq, self.reduced_trie[batch_id]
                    )
                else:
                    raise NotImplementedError("hmm something wrong")
            nex = self.get_from_trie(
                current_sequence, self.reduced_trie[batch_id], batch_id
            )
            self.lastnex[batch_id] = nex
            return nex

        nex = self.get_from_trie(prefix_sequence, self.trie_dict, batch_id)
        self.lastnex[batch_id] = nex
        return nex

    @staticmethod
    def load_from_dict(trie_dict: Dict):
        trie = Trie()
        trie.trie_dict = trie_dict
        trie.len = sum(1 for _ in trie)
        return trie

    @staticmethod
    def _add_to_trie(sequence: List[int], trie_dict: Dict):
        if sequence:
            if sequence[0] not in trie_dict:
                trie_dict[sequence[0]] = {}
            Trie._add_to_trie(sequence[1:], trie_dict[sequence[0]])

    def get_from_trie(
        self, prefix_sequence: List[int], trie_dict: Dict, bi: int
    ) -> List[int]:
        if prefix_sequence == [self.eos_token_id]:
            return [self.pad_token_id]
        elif len(prefix_sequence) == 0:
            return list(trie_dict.keys())
        elif prefix_sequence[0] in trie_dict:
            return self.get_from_trie(
                prefix_sequence[1:], trie_dict[prefix_sequence[0]], bi
            )
        elif (
            prefix_sequence[0] == self.pad_token_id
            or prefix_sequence[0] == self.bos_token_id
            or prefix_sequence[0] == self.eos_token_id
        ):
            return [self.pad_token_id]
        else:
            print("prefix_sequence", prefix_sequence)
            print("trie_dict", trie_dict)
            print("previous returned", self.lastnex[bi])
            print("last sequence", self.last_sequence[bi])
            raise ValueError("Trie Broken, sequence is not part of Trie")

    def delete(self, sequence: List[int], trie_dict: Dict) -> Dict:
        reduced_trie = deepcopy(trie_dict)
        if sequence[-1] == self.eos_token_id:
            sequence = sequence[:-1]
        self._delete_helper(sequence, reduced_trie)
        self.len -= 1
        return reduced_trie

    def _delete_helper(self, sequence: List[int], trie_dict: Dict):
        if not sequence:
            if self.sep_token_id in trie_dict:
                del trie_dict[self.sep_token_id]
            if self.eos_token_id in trie_dict:
                del trie_dict[self.eos_token_id]
            return

        token = sequence[0]
        if token in trie_dict:
            self._delete_helper(sequence[1:], trie_dict[token])
            if not trie_dict[token]:
                del trie_dict[token]

    def __iter__(self):
        def _traverse(prefix_sequence, trie_dict):
            if trie_dict:
                for next_token in trie_dict:
                    yield from _traverse(
                        prefix_sequence + [next_token], trie_dict[next_token]
                    )
            else:
                yield prefix_sequence

        return _traverse([], self.trie_dict)

    def __len__(self):
        return self.len

    def __getitem__(self, value):
        return self.get(value)
