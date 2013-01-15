#Driver for HW5

import random
import string
import subprocess
import json
import base64
import hashlib

#Magic Image String delimiters
image_start = 'BEGIN_IMAGE_f9825uweof8jw9fj4r8'
image_end   = 'END_IMAGE_0238jfw08fjsiufhw8frs'

timingStringIdentifier = 'e57__TIMING__f82'
meanStringIdentifier   = 'e57__MEAN__f82'
#strip all timing strings from the output
def stripPrints(inputString, identifier):
    pos = inputString.find(identifier)
    if pos == -1:
        return inputString, ''
    else:
        val = ''
        newOutputString = ''
        for line in inputString.split('\n'):
            if line.startswith(identifier):
                val = line.split()[1]
            elif not line == '':
                newOutputString += line + '\n'
        if identifier in newOutputString:
            #more than one!! bad news...probably cheating attempt
            return 'There is no reason to output the string ' + identifier + ' in your code\nCheating Suspected - Automatic Fail', ''
        else:
            return newOutputString, val

def runCudaAssignment():
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
    random_output_name = ''.join(random.choice(string.ascii_lowercase) for x in range(10)) + '.bin'

    #run their compiled code
    try:
        progOutput = subprocess.check_output(['./hw', random_output_name], stderr = subprocess.STDOUT)
    except subprocess.CalledProcessError, e:
        #program failed, dump possible Make warnings, program output and quit
        progOutput, time = stripPrints(e.output, timingStringIdentifier)
        progOutput, mean = stripPrints(progOutput, meanStringIdentifier)
        results['progOutput'] = progOutput
        print json.dumps(results)
        return

    progOutput, time = stripPrints(progOutput, timingStringIdentifier)
    progOutput, mean = stripPrints(progOutput, meanStringIdentifier)
    results['progOutput'] = progOutput
    results['time'] =       time

    #check if stripping timing or mean failed
    if time == '' or mean == '':
        print json.dumps(results)
        return

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
        results['compareOutput'] = subprocess.check_output(['./compare', random_output_name, mean], stderr = subprocess.STDOUT)
    except subprocess.CalledProcessError, e:
        results['compareOutput'] = e.output
        print json.dumps(results)
        return

    results['compareResult'] = True

    print json.dumps(results)

if __name__ == "__main__":
    runCudaAssignment()
