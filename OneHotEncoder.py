class OneHotCharacter:
    def __init__(self, set_of_chars):
        self.end_seq_str = "_END_"

        set_of_chars.sort()

        # add to set_of_chars string that represents end of sequence
        set_of_chars.append(self.end_seq_str)

        self.__vec_to_char = {
            tuple([0 if i != set_of_chars.index(char) else 1 for i in range(len(set_of_chars))]): char for char in
            set_of_chars}

        # reverse
        self.__char_to_vec = {v: k for k, v in self.__vec_to_char.items()}

        # set end_seq_vec
        self.end_seq_vec = self.__char_to_vec[self.end_seq_str]

    def __getitem__(self, item):
        if isinstance(item, tuple):  # given vector
            return self.__vec_to_char[item]
        else:
            return self.__char_to_vec[item]

    def __len__(self):
        return len(self.__vec_to_char)