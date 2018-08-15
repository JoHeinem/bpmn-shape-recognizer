class IdGenerator(object):

    number = 0

    @staticmethod
    def next():
        tmp = IdGenerator.number
        IdGenerator.number += 1
        return str(tmp)
