from sys import argv
import random


def main(argv):
	assert(len(argv) == 4)
	numPoints = int(argv[1])
	patchSize = int(argv[2])
	fileName = argv[3]

	points = [0] * numPoints
	i = 0
	patchWidth = (patchSize - 1) / 2
	while (i < numPoints):
		x1 = random.randint(-patchWidth, patchWidth) 
		y1 = random.randint(-patchWidth, patchWidth)
		x2 = random.randint(-patchWidth, patchWidth)
		y2 = random.randint(-patchWidth, patchWidth)
		newPoint = (x1, y1, x2, y2)
		alreadyThere = False
		for point in points:
			if (point == newPoint):
				alreadyThere = True
				break
		if (alreadyThere): continue
		points[i] = (x1, y1, x2, y2)
		i += 1

	with open(fileName, "w+") as f:
		f.write("{}\n".format(str(numPoints)))
		for (a, b, c, d) in points:
			message = "{} {} {} {}\n".format(a, b, c, d)
			f.write(message)

if __name__ == '__main__':
	main(argv)