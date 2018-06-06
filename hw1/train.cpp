#include "hmm.h"
#include <math.h>
#include <string>
#include <vector>
#include <iostream>
using namespace std;
//#define DEBUG
//time ./train 1200 model_init.txt seq_model_01.txt model_01.txt
int main(int argc, char* argv[]){
    int iters = atoi(argv[1]);

    HMM model;
    loadHMM(&model, argv[2]);

    /* loading seq_model */
    vector<string> seq_model;
    FILE *fp = fopen(argv[3], "r");
    char buf[MAX_SEQ], *re_val; string buf_str;
    while(1){
        re_val = fgets(buf, sizeof(buf), fp);
        if(feof(fp)) break;
        buf[strlen(buf) - 1] = 0;
        buf_str = buf;
        seq_model.push_back(buf_str);
    }

    /* Training */
    for(int iter=0; iter<iters; iter++){
        int N = model.state_num;
        #ifdef DEBUG
        if(iter % 50 == 0) printf("iter = %d\n", iter);
        #endif

        double A[MAX_STATE][MAX_SEQ] = {};
        double B[MAX_STATE][MAX_SEQ] = {};
        double G[MAX_STATE][MAX_SEQ] = {};
        double E[MAX_STATE][MAX_STATE][MAX_SEQ] = {};
    
        double sumG[MAX_STATE] = {};
        double a_sumE[MAX_STATE][MAX_STATE] = {};
        double a_sumG[MAX_STATE] = {};
        double b_sumOG[MAX_STATE][MAX_STATE] = {};
        double b_sumG[MAX_STATE] = {};

        for(int line=0; line<seq_model.size(); line++){     //for line in seq_model
            string seq = seq_model[line];
            int T = seq.length();
            double S = 0;   //S: sigma

            /* α(A) Forward Procedure */
            for (int i=0; i<N; i++)
                A[i][0] = model.initial[i] * model.observation[seq[0] - 'A'][i];
            for (int t=1; t<T; t++){
                for (int i=0; i<N; i++){
                    S = 0;
                    for (int j=0; j<N; j++) S += A[j][t-1] * model.transition[j][i];
                    A[i][t] = model.observation[seq[t] - 'A'][i] * S;
                }
            }

            /* β(B) Backward procedure */
            for (int i=0; i<N; i++) B[i][T-1] = 1;
            for (int t=T-2; t>=0; t--){
                for (int i=0; i<N; i++){
                    B[i][t] = 0;
                    for (int j=0; j<N; j++) B[i][t] += B[j][t+1] *
                        model.transition[i][j] * model.observation[seq[t+1] - 'A'][j];
                }
            }

            /* Calculate γ(G) */
            for (int t=0; t<T; t++){
                S = 0;
                for (int i=0; i<N; i++) S += A[i][t] * B[i][t];
                for (int i=0; i<N; i++) G[i][t] = A[i][t] * B[i][t] / S;              
            }

            /* Calculate ε(E) */
            for (int t=0; t<(T-1); t++){  
                S = 0;
                for (int i=0; i<N; i++){
                    for (int j=0; j<N; j++) S += A[i][t] * model.transition[i][j] * 
                        B[j][t + 1] * model.observation[seq[t + 1] - 'A'][j];
                }
                for (int i=0; i<N; i++){
                    for (int j=0; j<N; j++) E[i][j][t] = (A[i][t] * model.transition[i][j] * 
                            B[j][t + 1] * model.observation[seq[t + 1] - 'A'][j]) / S;
                }
            }

            /* Accumulate ε and γ */
            for (int i=0; i<N; i++){
                sumG[i] += G[i][0];
                for (int t=0; t<(T - 1); t++){
                    a_sumG[i] += G[i][t];
                    for (int j=0; j<N; j++) a_sumE[i][j] += E[i][j][t];
                }
                for (int t=0; t<T; t++){
                    b_sumG[i] += G[i][t];
                    b_sumOG[seq[t] - 'A'][i] += G[i][t];
                }
            }
        }

        /* Update parameters */
        for (int i=0; i<N; i++) model.initial[i] = sumG[i] / seq_model.size();
        for (int i=0; i<N; i++){
            for (int j=0; j<N; j++){
                model.transition[i][j] = a_sumE[i][j] / a_sumG[i];
                model.observation[j][i] = b_sumOG[j][i] / b_sumG[i];
            }
        }

        #ifdef DEBUG
        /* Dumping output */
        char outpath[50], listpath[50];
        sprintf(outpath, "models/%d_%s", iter, argv[4]);
        sprintf(listpath, "models/%d_modellist.txt", iter);
        FILE *listfile = fopen(listpath, "a+");
        FILE *outfile = open_or_die(outpath, "w");
        fprintf(listfile, "%s\n", outpath);
        dumpHMM(outfile, &model);
        fclose(listfile); fclose(outfile); 
        #endif
    }

    /* Dumping output */
    dumpHMM(open_or_die(argv[4], "w"), &model);
    return 0;
}