import numpy as np 
class Trie:
    def __init__(self, bags_of_words):
        self.root = Trie_Node()
        self.count_vector = np.zeros(len(bags_of_words))
        self.index = {}
        i = 0
        for item in bags_of_words:
            self.index[item] = i
            i += 1
            self.root.add(item, i)

    def contains(self, word, frequency = True):
        index = self.root.contains(word)
        if index != -1:
            if frequency:
                self.count_vector[index] += 1
            else:
                self.count_vector[index] = 1
        return self.count_vector

class Trie_Node:
    def __init__(self):
        self.children = [None] * 27
        self.character = [False] * 27
        self.isLeaf = [-1] * 27

    def add(self, word, index):
        if word[0] == ' ':
            index = 26
        else:
            index = ord(word[0]) - ord('a')
        self.character[index] = True
        if len(word) == 1:
            self.isLeaf[index] = index
        else:
            if not self.children[index]:
                self.children[index] = Trie_Node()
            self.children[index].add(word[1:], index)
    def contains(word):
        if word[0] == ' ':
            index = 26
        else:
            index = ord(word[0]) - ord('a')
        if not self.character[index]:
            return -1
        else:
            if self.isLeaf[index] != -1:
                return self.isLeaf[index]
            else:
                return self.children[index].contains[word[1:]]
