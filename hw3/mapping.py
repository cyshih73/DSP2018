import sys

dictionary = {}
# Input
input_file = open(sys.argv[1], "r", encoding='big5hkscs')
for line in input_file:
    line = line.split(' ')
    line[1] = line[1].strip().split('/')

    for x in line[1]:
        if x[0] not in dictionary:
            dictionary[x[0]] = set()
        dictionary[x[0]].add(line[0])
    dictionary[line[0]] = [line[0]]

# Output
file_output = open(sys.argv[2], 'w+', encoding='big5hkscs') 
for k, v in dictionary.items():
    print('{}\t{}'.format(k, ' '.join(list(v))), file=file_output)
