#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <omp.h>


int main(int argc, char *argv[]) {

	char *N = argv[1];
	char *t = argv[2];
	int num = atoi(N);
	int thread = atoi(t);
	int stop = (num + 1) / 2;

	int prime[num - 1];
	double tstart = 0.0, ttaken;

	tstart = omp_get_wtime();
	#pragma omp parallel num_threads (thread)
	{

		#pragma omp for
		for (int i = 2; i <= num; i++) {
			prime[i-2] = i;
		}

		#pragma omp for 
		for (int j = 2; j <= stop; j++) {
			
			if (prime[j-2] == 0) {
				continue;
			
			}

			for (int k = j-1; k < num-1; k++) {
				if (prime[k] % j == 0) {
					prime[k] = 0;
				}
			}
		}	
	}
	ttaken = omp_get_wtime() - tstart;
	printf("Time take for the main part: %f\n", ttaken);

	FILE *fp;
	fp = fopen ("N.txt", "w");
	int prevprime = 2;
	int index = 1;
	for (int k = 0; k < num-1; k++) {
		if (prime[k] != 0) {
			fprintf(fp, "%d, %d, %d\n", index, prime[k], prime[k]-prevprime);
			index += 1;
			prevprime = prime[k];
		}
	}
	fclose(fp);
}
