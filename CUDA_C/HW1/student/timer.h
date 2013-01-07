#ifndef TIMER_H__
#define TIMER_H__

#include <sys/time.h>
#include <cuda_runtime.h>

//very simple timer that just returns elapsed time
//since the last call
double tick(void) {
  static unsigned long long prev_usecs = 0;

  cudaDeviceSynchronize();
  timeval tv;
  gettimeofday(&tv, NULL);
  unsigned long long curr_usecs = tv.tv_usec + 1000000L * tv.tv_sec;

  if (prev_usecs) {
    double msecs_elapsed = (curr_usecs - prev_usecs) / 1000.;
    prev_usecs = curr_usecs;
    return msecs_elapsed;
  }
  else {
    prev_usecs = curr_usecs;
    return 0;
  }
}

#endif
