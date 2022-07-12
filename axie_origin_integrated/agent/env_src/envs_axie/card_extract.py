import json
import yaml

f = open('data.json')

data = json.load(f)

# 'cards', 'tool_cards'

# print(data.keys())
# print(len(data['cards']))
# print(len(data['tool_cards']))

name_set = set()

duplicate_card = ['Anemone', 'Nimo', 'Puppy', 'Foxy', 'Nut Cracker', 'Puff', 'Nut Cracker', 'Little Owl', 'Peace Maker', 'Little Owl', 'Leaf Bug', 'Kotaro', 'Tiny Dino']

for card in data['cards']:
    card_name = card['name']
    if (card_name in duplicate_card):
        name_set.add(card_name + '(' + card['part_type'] + ')')
    else:
        name_set.add(card_name)

for card in data['tool_cards']:
    card_name = card['name']
    if (card_name in duplicate_card):
        name_set.add(card_name + '(' + card['part_type'] + ')')
    else:
        name_set.add(card_name)

names = []
for x in name_set:
    if (type(x) == str):
        names.append(x)

names.sort()

dict = {}
current_index = 0
for name in names:
    dict[name] = current_index
    current_index += 1

with open(r'config/card2id.yaml', 'w') as file:
    document = yaml.dump(dict, file)