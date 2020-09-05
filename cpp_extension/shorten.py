# C++ のコードを 1 行にする

import re
def comment_remover(text):
    # from https://stackoverflow.com/questions/241327/remove-c-and-c-comments-using-python
    def replacer(match):
        s = match.group(0)
        if s.startswith('/'):
            return " " # note: a space and not an empty string
        else:
            return s
    pattern = re.compile(
        r'//.*?$|/\*.*?\*/|\'(?:\\.|[^\\\'])*\'|"(?:\\.|[^\\"])*"',
        re.DOTALL | re.MULTILINE
    )
    return re.sub(pattern, replacer, text)

def shorten(text):
    text = comment_remover(text)
    return re.sub(r" +", " ", text).replace("\n ", "\n").replace("\n", r"\n")
