#Sample python file for running CUDA Assignments
#Created 1/6/13 by Erich Elsen

import StringIO
import random
import string
import subprocess
import json
import base64


def runCudaAssignment():
    stdOutput = StringIO.StringIO()

    #call make, capture stderr & stdout
    #if make fails, quit and dump errors to student
    try:
        stdOutput.write(subprocess.check_output(['make'], stdout = stdOutput, stderr = subprocess.STDOUT))
    except subprocess.CalledProcessError, e:
        #output make error(s)
        print e.output
        return

    #generate a random output name so that students cannot copy the gold file
    #directly to the output file
    random_output_name = ''.join(random.choice(string.ascii_lowercase) for x in range(10)) + '.png'

    #run their compiled code
    try:
        stdOutput.write(subprocess.check_output(['./hw', 'cinque_terre_small.jpg', random_output_name], stdout = stdOutput, stderr = subprocess.STDOUT))
    except subprocess.CalledProcessError, e:
        #program failed, dump possible Make warnings, program output and quit
        print stdOutput.getvalue(), e.output
        return

    #gather timing information from stdOutput
    foundTiming = False
    for line in stdOutput.getvalue().split('\n'):
        if line.startswith('TIMING'): #TODO this string is not set yet
            if foundTiming:
                print 'Do not add your own print beginning with TIMING!!'
                print stdOutput.getvalue()
                return
            foundTiming = True
            time = line.split()[5] #TODO making this index up right now

    print 'time', time

    #We should've found a timing statement
    if not foundTiming:
        print 'Something bad has happened, timing statement should have been found'
        print stdOutput.getvalue()
        return

    #run compare to see if the output is correct
    try:
        stdOutput.write(subprocess.check_output(['./compare', 'cinque_terre.gold', random_output_name], stdout = stdOutput, stderr = subprocess.STDOUT))
    except subprocess.CalledProcessError, e:
        #images don't match
        #dump image anyway?
        print stdOutput.getValue(), e.output
        return

    #everything looks good, open image and return it with magic strings
    imageFile = open(random_output_name, 'rb').read()

    #add magic string, JSONize and dump to stdout as well
    image_start = 'BEGIN_IMAGE_f9825uweof8jw9fj4r8'
    image_end   = 'END_IMAGE_0238jfw08fjsiufhw8frs'

    stdOutput.write(image_start)

    data = {}
    data['name'] = 'StudentImage'
    data['format'] = 'png'
    data['bytes'] = base64.encodestring(imageFile)

    stdOutput.write(json.dumps(data) + image_end)

    #now actually write everything to stdout
    print stdOutput.getvalue()

if __name__ == "__main__":
    runCudaAssignment()
