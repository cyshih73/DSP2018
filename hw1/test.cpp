#include "hmm.h"
#include <math.h>
#include <string>
#include <vector>
#include <algorithm>
#include <iostream>
using namespace std;

#define MAX_MODEL 10
//./test modellist.txt testing_data1.txt result1.txt
double viterbi(HMM hmm, string seq){
    double D[2][MAX_STATE] = {};     //delta
    double *P;
    int N = hmm.state_num, T = seq.length();
    
    for (int i=0; i<N; i++) 
        D[0][i] = hmm.initial[i] * hmm.observation[seq[0] - 'A'][i];

    for (int t=1; t<T; t++){
        for (int j=0; j<N; j++){
            double temp_D = 0;
            for (int i=0; i<N; i++)
                if(D[(t+1)%2][i] * hmm.transition[i][j] > temp_D)
                    temp_D = D[(t+1)%2][i] * hmm.transition[i][j];
            D[t%2][j] = temp_D * hmm.observation[seq[t]-'A'][j];
        }
        P = D[t%2];
    }

    return *max_element(P, P + N);  
}

int main(int argc, char *argv[]){
	HMM hmms[MAX_MODEL];
	int count_model = load_models(argv[1], hmms, MAX_MODEL);

    vector<string> seq_model;
    FILE *fp = open_or_die(argv[2], "r");
    char buf[MAX_SEQ], *re_val; string buf_str;
    while(1){
        re_val = fgets(buf, sizeof(buf), fp);
        if(feof(fp)) break;
        buf[strlen(buf) - 1] = 0;
        buf_str = buf;
        seq_model.push_back(buf_str);
    }

    fp = open_or_die(argv[3], "w");
    for(int line=0; line<seq_model.size(); line++){
        string seq = seq_model[line];
        int model_id;
        double P_max = 0;
        for (int i=0; i<count_model; i++){
            double value = viterbi(hmms[i], seq);
            if (value > P_max){ P_max = value; model_id = i; }
        }
        fprintf(fp, "%s %g\n", hmms[model_id].model_name, P_max);
    }

    fclose(fp);
    return 0;
}
