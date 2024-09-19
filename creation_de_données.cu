#include <stdio.h>

typedef unsigned int uint;
//	= fread() & fwrite() sans les signaux de -Wall ou -O3 =
FILE * FOPEN(char * fichier, char * mode);
#define FREAD(ptr, taille, nb, fp) (void)!fread(ptr, taille, nb, fp);
#define FWRITE(ptr, taille, nb, fp) (void)!fwrite(ptr, taille, nb, fp);
#define SI_EXISTE(fp, fichier) do {if (fp == 0) ERR("Fichier %s existe pas", fichier);} while(0);
#define FOPEN_LOCK(fp, fichier) do {int fd = fileno(fp);flock(fd, LOCK_EX);} while(0);
#define FCLOSE_UNCLOCK(fp) do {flock(fileno(fp), LOCK_UN);fclose(fp);} while(0);
#define FOR(d,i,N)       for (uint i=d  ; i <  N; i++)

int main() {
	FILE * fp = fopen("instructions_rapiditée", "rb");
	//
	uint informations;
	uint  extractions;
	uint            T;
	uint            N;
	uint       DEPART;
	//
	FREAD(&informations, sizeof(uint), 1, fp);
	FREAD(& extractions, sizeof(uint), 1, fp);
	FREAD(&           T, sizeof(uint), 1, fp);
	FREAD(&           N, sizeof(uint), 1, fp);
	FREAD(&      DEPART, sizeof(uint), 1, fp);
	//
	float * informations_brute[informations];
	FOR(0, i, informations) {
		informations_brute[i] = (float*)malloc(sizeof(float) * T);
		FREAD(informations_brute[i], sizeof(float), T, fp);
	}
	//
	uint MODELES;
	FREAD(&    MODELES, sizeof(uint), 1, fp);
	//
	float * emaA[MODELES*informations*extractions];
	float * emaB[MODELES*informations*extractions];
	//
	uint _I0[MODELES*informations*extractions];
	uint _I1[MODELES*informations*extractions];
	//
	FOR(0, m, MODELES) {
		FOR(0, i, informations) {
			FOR(0, e, extractions) {
				uint K0, K1, I0, I1;
				FREAD(&K0, sizeof(uint), 1, fp);
				FREAD(&K1, sizeof(uint), 1, fp);
				FREAD(&I0, sizeof(uint), 1, fp);
				FREAD(&I1, sizeof(uint), 1, fp);
				//
				uint ie = m*informations*extractions + i*extractions + e;
				//
				_I0[ie] = I0;
				_I1[ie] = I1;
				//
				emaA[ie] = (float*)malloc(sizeof(float) * T);
				emaB[ie] = (float*)malloc(sizeof(float) * T);
				//
				emaA[ie][0] = informations_brute[i][0];
				emaB[ie][0] = informations_brute[i][0];
				FOR(1, t, T) {
					emaA[ie][t] = emaA[ie][t-1] * (1-1/(float)K0) + informations_brute[i][t]/(float)K0;
					emaB[ie][t] = emaB[ie][t-1] * (1-1/(float)K1) + informations_brute[i][t]/(float)K1;
				}
			}
		}
	}
	//
	fclose(fp);

	//	==================================================================================
	//	==================================================================================
	//	==================================================================================

	FOR(0, m, MODELES) {
		char fichier[100];
		snprintf(fichier, 100, "X_bloques_par_mdl_%i", m);
		//
		fp = fopen(fichier, "wb");
		//
		FOR(DEPART, t, T) {
			FOR(0, ie, informations*extractions) {
				FOR(0, n, N) {
					float v = emaA[ie][ t - n*_I0[ie] ] / emaB[ie][ t - n*_I0[ie]*_I1[ie] ]  -  1;
					FWRITE(&v, sizeof(float), 1, fp);
				}
			}
		}
		//
		fclose(fp);
	}
	//
	printf("[./creation_de_données] Fin d'écriture de bloque\n");
}