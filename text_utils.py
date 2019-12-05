PATH = ''
DATA = ''
WORDS = ''
SEED_PAIRS = ''
RANKED_PAIRS = ''
MODEL = ''

tweet_delim = '======================\n'
urls = [r'https?:\/\/(www\.)?[-a-zA-Z0-9@:%._\+~#=]{1,256}\.[a-zA-Z0-9()]{1,6}\b([-a-zA-Z0-9()@:%_\+.~#?&//=]*)',
        r'[-a-zA-Z0-9@:%._\+~#=]{1,256}\.[a-zA-Z0-9()]{1,6}\b([-a-zA-Z0-9()@:%_\+.~#?&//=]*)',
           r'status/.* ',
           r'us/.* ',
           r'https://',
           r'http://',
           r'www.']

punct = r'#@$&?.<> \n\t'

hash_tag = r'#[\w\d]+'
handle_tag = r'@[\w\d]+'
'''
abbreviations = ['AAP', 'AUS', 'BAN', 'BJP', 'CWC', 'ENG', 'INC', 'IND', 
                 'JNU', 'NCP', 'NDA', 'PAK', 'RBI', 'SRK', 'TV', 'UPA']
'''
handle_sub = '<hndl>'