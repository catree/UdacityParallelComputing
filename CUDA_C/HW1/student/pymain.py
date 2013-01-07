#Sample python file for running CUDA Assignments
#Created 1/6/13 by Erich Elsen

import StringIO
import random
import string
import subprocess
import json
import base64
import hashlib

#strip all timing strings from the output
def stripTimingPrints(inputString):
    timingStringIdentifier = 'e57__TIMING__f82'
    pos = inputString.find(timingStringIdentifier)
    if pos == -1:
        return inputString, ''
    else:
        for line in inputString.split('\n'):
            if line.startswith(timingStringIdentifier):
                time = line.split()[3]
        inputString = inputString[0:pos] + inputString[pos + len(timingStringIdentifier):]
        if timingStringIdentifier in inputString:
            #more than one!! bad news...probably cheating attempt
            return 'There is no reason to output the string ' + timingStringIdentifier + ' in your code\nCheating Suspected - Automatic Fail', ''
        else:
            return inputString, time

def runCudaAssignment():
    stdOutput = StringIO.StringIO()
    results = {'Makefile':'', 'progOutput':'', 'compareOutput': '', 'compareResult':False, 'time':''}

    #call make, capture stderr & stdout
    #if make fails, quit and dump errors to student
    try:
        results['Makefile'] = subprocess.check_output(['make', '-s'], stderr = subprocess.STDOUT)
    except subprocess.CalledProcessError, e:
        #output make error(s)
        results['Makefile'] = e.output
        print json.dumps(results)
        return

    
    #generate a random output name so that students cannot copy the gold file
    #directly to the output file
    random_output_name = ''.join(random.choice(string.ascii_lowercase) for x in range(10)) + '.png'

    #run their compiled code
    try:
        progOutput = subprocess.check_output(['./hw', 'cinque_terre_small.jpg', random_output_name], stderr = subprocess.STDOUT)
    except subprocess.CalledProcessError, e:
        #program failed, dump possible Make warnings, program output and quit
        progOutput, time = stripTimingPrints(e.output)
        results['progOutput'] = e.output
        print json.dumps(results)
        return

    progOutput, time = stripTimingPrints(progOutput)
    results['progOutput'] = progOutput
    results['time'] =       time

    #run compare to see if the output is correct
    #get md5 hash of compare executable and check with known result to avoid tampering...
    try:
        hashVal = hashlib.sha1(open('./compare', 'rb').read()).hexdigest()
    except IOError, e:
        #probably couldn't open compare - a bad sign
        results['compare'] = e
        print json.dumps(results)
        return
 
    #Uncomment these lines once the SHA1 value of the compare executable is known

    #goldHashVal = '?' #TODO needs to be determined for server (ideally precompiled executable)

    #if not hashVal == goldHashVal:
    #    results['compare'] = 'Compare executable has been modified!'
    #    print json.dumps(results)
    #    return

    try:
        results['compareOutput'] = subprocess.check_output(['./compare', 'cinque_terre.gold', random_output_name], stderr = subprocess.STDOUT)
    except subprocess.CalledProcessError, e:
        #images don't match
        #dump image anyway?
        results['compareOutput'] = e.output
        print json.dumps(results)
        return

    results['compareResult'] = True

    print json.dumps(results)
    #everything looks good, open image and return it with magic strings
    imageFile = open(random_output_name, 'rb').read()

    #add magic string, JSONize and dump to stdout as well
    image_start = 'BEGIN_IMAGE_f9825uweof8jw9fj4r8'
    image_end   = 'END_IMAGE_0238jfw08fjsiufhw8frs'

    data = {}
    data['name'] = 'StudentImage'
    data['format'] = 'png'
    data['bytes'] = base64.encodestring(imageFile)

    print image_start + json.dumps(data) + image_end


if __name__ == "__main__":
    runCudaAssignment()
