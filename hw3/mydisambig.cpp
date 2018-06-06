#include <cstdio>
#include <cstring>
#include <limits>

#include "File.h"
#include "Prob.h"
#include "Ngram.h"
#include "Vocab.h"
#include "VocabMap.h"
#include "Trellis.cc"
#include "TextStats.h"

static char arg_text[128], arg_lm[128], arg_map[128];
static unsigned arg_order;

void arg_parse(int argc, char* argv[]){
    for(int i=0; i<argc; i++){
        if(!strcmp(argv[i], "-text")) strcpy(arg_text, argv[i+1]);
        if(!strcmp(argv[i], "-lm")) strcpy(arg_lm, argv[i+1]);
        if(!strcmp(argv[i], "-map")) strcpy(arg_map, argv[i+1]);
        if(!strcmp(argv[i], "-order")) sscanf(argv[i+1], "%u", &arg_order);
    }
    return;
}

int main(int argc, char* argv[]){
    // Parse the arguments
    arg_parse(argc, argv);
    File file_text(arg_text, "r");
    File file_lm(arg_lm, "r");
	File file_map(arg_map, "r");

    // Read provided files
    Vocab ZhuYin, Big5;
	VocabMap map(ZhuYin, Big5);
    map.read(file_map); file_map.close();
    Ngram lm(Big5, arg_order);
    lm.read(file_lm); file_lm.close();

    // Start recognizing
    char* line;
    while( line = file_text.getline() ) {
		// Initialization; (maxWordsPerLine defined in File.h)
        VocabString words[maxWordsPerLine];
		VocabIndex idx_word[maxWordsPerLine + 2];
		unsigned length_line = Vocab::parseWords(line, words, maxWordsPerLine) + 2;

        map.vocab1.getIndices(words, &idx_word[1], maxWordsPerLine, map.vocab1.unkIndex());
        idx_word[0] = Big5.getIndex("<s>"); // <s>
        idx_word[length_line - 1] = Big5.getIndex("</s>"); // </s>
        idx_word[length_line] = Vocab_None; // End

		// trellis: Vector liked structure
    	Trellis <const VocabIndex*> trellis(length_line);
		VocabMapIter iter(map, idx_word[0]);
        VocabIndex idx_temp[2] = {Vocab_None};
        Prob prob_temp;
        while(iter.next(idx_temp[0], prob_temp)) 
			trellis.setProb(idx_temp, ProbToLogP(prob_temp));

		// Find candidates
		unsigned pos = 1;
		const VocabIndex idx_none[] = {Vocab_None};
		while (idx_word[pos] != Vocab_None){
			trellis.step();
			VocabMapIter iter_pos(map, idx_word[pos]);
			
			while (iter_pos.next(idx_temp[0], prob_temp)){
				VocabIndex idx_new[maxWordsPerLine + 2];
				idx_new[0] = idx_temp[0];
				LogP prob_Gram = lm.wordProb(idx_temp[0], idx_none);
				LogP prob_Local = ProbToLogP(prob_temp);
				TrellisIter<const VocabIndex*> iter_old(trellis, pos - 1);
				const VocabIndex* idx_old;
				LogP prob_old;
				
				// Update
				while (iter_old.next(idx_old, prob_old)) {
					LogP prob_transition = lm.wordProb(idx_temp[0], idx_old);
					if (prob_transition == LogP_Zero && prob_Gram == LogP_Zero) 
						prob_transition = -(std::numeric_limits<float>::max());
					idx_new[1] = Vocab_None;
					trellis.update(idx_old, idx_new, prob_transition + prob_Local);
				}
			}
			pos++;
		}
		// Viterbi
		const VocabIndex* idx_hmm[length_line + 1];
		trellis.viterbi(idx_hmm, length_line);
		
		// Output
		for(int i = 0; i < length_line-1; i++) 
			printf("%s ", map.vocab2.getWord(idx_hmm[i][0]));
		printf("</s>\n");
	}
    file_text.close();
    return 0;
}
