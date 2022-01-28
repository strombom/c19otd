
import dateutil.parser


transcript = open('transcript.txt', mode="r", encoding="utf-8").read()

pairs = []

pair = []

c = 0

for row in transcript.split('\n'):
	try:
		date = dateutil.parser.isoparse(row.replace('.000', '').replace('.', ':'))
		# New date
		if pair:
			pairs.append(pair)
		pair = []
		continue
	except:
		pass


	print(date)
	#try:


	c = c + 1
	if c == 10:
		quit()


print(transcript)

