#include <iostream>
#include <time.h>
#include <fstream>
#include <random>
#include <chrono>
#include <vector>
#include <algorithm>
#include <numeric>
#include <cmath>
#include <iomanip>
#include <filesystem>
#include <string>
#include <limits>

const int Benchmark = 17; // 17 (CEC 2017 and CEC 2024) or 22 (CEC 2022)
//#include "cec22_test_func.cpp"
#include "cec17_test_func.cc"
//#include "cec17_test_fast_pow.cpp" // Should work faster

//Ctrl+F "CHANGE" in the file and replace 17 with 22
const int ResTsize1 = 29; // number of functions //12 for CEC 2022, 29 for CEC 2024(2017)
const int ResTsize2 = 1001; // number of records per function //1000+1 = 1001 for CEC 2024
const int NUM_RUNS = 25;

// =====================
// Phase 1 / Phase 2 controls
// =====================
// CHANGE: keep one clear experiment identity so baseline and future candidates
// can be compared cleanly with separate output folders and files.
// const char* algName = "RDEx";
// const char* variantName = "lsv1_s75sf025_xi075k75";
// const char* experimentTag = "p12";
const char* algName = "RDEx";
const char* variantName = "lsv1_s75sf025_xi065k65";
const char* experimentTag = "p12";

// CHANGE: keep all outputs for a given experiment in a dedicated folder.
const char* outputRootDir = "RDEx_results";

// CHANGE: use the same run seeds for baseline and future candidates.
// This is the most important step for fair A/B comparisons.
const bool USE_FIXED_RUN_SEEDS = true;
const unsigned MASTER_SEED = 20260417u;

// CHANGE: generation-level diagnostics. To avoid huge files, diagnostics are
// saved only for selected run indices by default.
const bool SAVE_DIAGNOSTICS = true;
const int DIAGNOSTIC_RUN_INDEX = 0;

// =====================
// CHANGE (lsv1_start75_sigmaF025_xi072): late smooth-only winner + two scalar refinements
// =====================
// This candidate keeps the winning late-only smooth control:
//   - hard late start at 0.75
//   - smoothing weight 0.65
//   - no clamping
// and adds two additional one-scalar refinements:
//   - sigmaF: 0.02 -> 0.025
//   - xi coefficient in psizeval: 0.70 -> 0.72
const bool ENABLE_LATE_EB_CONTROL = true;
const double LATE_EB_ENABLE_PROGRESS_FRACTION = 0.75;
const double LATE_EB_SMOOTH_OLD_WEIGHT = 0.65;   // keep 65% of previous EB rate

// CHANGE (lsv1_start75_sigmaF025_xi072): additional one-scalar refinements
const double STANDARD_BRANCH_SIGMA_F = 0.025;
const double PSIZEVAL_XI_COEFF = 0.65;
const double PSIZEVAL_k_COEFF = 6.5; // new clamp coefficient for psizeval exponential decay, tuned in pilot experiments
const double EB_PSIZEVAL2_COEFF = 0.17;   

// Original RDEx hyperparameter retained unchanged.
double EB_hybrid_rate_init = 0.7;

using namespace std;
/*typedef std::chrono::high_resolution_clock myclock; // for random seed
myclock::time_point beginning = myclock::now();
myclock::duration d1 = myclock::now() - beginning;
#ifdef __linux__
    unsigned globalseed = d1.count();
#elif _WIN32
    unsigned globalseed = unsigned(time(NULL));
#else

#endif*/

// CHANGE: keep RNG state global, but provide a reset function so each run can
// start from a known seed when benchmarking baseline vs candidate versions.
unsigned globalseed = unsigned(time(NULL));
unsigned seed1 = globalseed;
unsigned seed2 = globalseed+100;
unsigned seed3 = globalseed+200;
unsigned seed4 = globalseed+300;
unsigned seed5 = globalseed+400;
std::mt19937 generator_uni_i(seed1);
std::mt19937 generator_uni_r(seed2);
std::mt19937 generator_norm(seed3);
std::mt19937 generator_uni_i_3(seed4);
std::mt19937 generator_cachy(seed5);
std::uniform_int_distribution<int> uni_int(0,32768);
std::uniform_real_distribution<double> uni_real(0.0,1.0);
std::normal_distribution<double> norm_dist(0.0,1.0);
std::cauchy_distribution<double> cachy_dist(0.0, 1.0);

void SetGlobalSeed(unsigned baseSeed)
{
    globalseed = baseSeed;
    seed1 = globalseed;
    seed2 = globalseed + 100;
    seed3 = globalseed + 200;
    seed4 = globalseed + 300;
    seed5 = globalseed + 400;
    generator_uni_i.seed(seed1);
    generator_uni_r.seed(seed2);
    generator_norm.seed(seed3);
    generator_uni_i_3.seed(seed4);
    generator_cachy.seed(seed5);
    srand(globalseed + 500);
}

int IntRandom(int target) {if(target == 0) return 0; return uni_int(generator_uni_i)%target;}
double Random(double minimal, double maximal){return uni_real(generator_uni_r)*(maximal-minimal)+minimal;}
double NormRand(double mu, double sigma){return norm_dist(generator_norm)*sigma + mu;}
double CachyRand(double mu, double sigma) {return cachy_dist(generator_cachy) * sigma + mu;}

double *OShift,*M,*y,*z,*x_bound;
int ini_flag=0,n_flag,func_flag,*SS;
int stepsFEval[ResTsize2-1];
double ResultsArray[ResTsize2];
double ResultsArray2[NUM_RUNS][1001];
int LastFEcount;
int NFEval = 0;
int MaxFEval = 0;
int GNVars;
double tempF[1];
double fopt;
char buffer[500];
double globalbest;
bool globalbestinit;
bool TimeComplexity = true;

double ComputeMean(const std::vector<double>& values)
{
    if(values.empty()) return 0.0;
    double sum = std::accumulate(values.begin(), values.end(), 0.0);
    return sum / static_cast<double>(values.size());
}

double ComputeStdSample(const std::vector<double>& values, double mean)
{
    if(values.size() <= 1) return 0.0;
    double accum = 0.0;
    for(double v : values)
    {
        double d = v - mean;
        accum += d * d;
    }
    return std::sqrt(accum / static_cast<double>(values.size() - 1));
}

double ComputeMedian(std::vector<double> values)
{
    if(values.empty()) return 0.0;
    std::sort(values.begin(), values.end());
    size_t n = values.size();
    if(n % 2 == 1) return values[n / 2];
    return 0.5 * (values[n / 2 - 1] + values[n / 2]);
}



std::string BuildExperimentPrefix()
{
    return std::string(algName) + "_" + variantName + "_" + experimentTag;
}

std::filesystem::path BuildExperimentDir()
{
    return std::filesystem::path(outputRootDir) / BuildExperimentPrefix();
}

std::vector<unsigned> BuildRunSeeds(int totalRuns)
{
    std::vector<unsigned> runSeeds(totalRuns);
    if(USE_FIXED_RUN_SEEDS)
    {
        std::mt19937 seeder(MASTER_SEED);
        std::uniform_int_distribution<unsigned> seedDist(1u, std::numeric_limits<unsigned>::max() - 1000u);
        for(int i = 0; i < totalRuns; ++i)
            runSeeds[i] = seedDist(seeder);
    }
    else
    {
        unsigned base = static_cast<unsigned>(time(NULL));
        for(int i = 0; i < totalRuns; ++i)
            runSeeds[i] = base + static_cast<unsigned>(1000 * (i + 1));
    }
    return runSeeds;
}

// CHANGE: reset the stored checkpoint values before each run. This makes the
// benchmark harness reliable and avoids cross-run contamination.
void ResetResultsArray()
{
    for(int i = 0; i < ResTsize2 - 1; ++i)
        ResultsArray[i] = std::numeric_limits<double>::quiet_NaN();
    ResultsArray[ResTsize2 - 1] = MaxFEval;
}

void qSort2int(double* Mass, int* Mass2, int low, int high)
{
    int i=low;
    int j=high;
    double x=Mass[(low+high)>>1];
    do
    {
        while(Mass[i]<x)    ++i;
        while(Mass[j]>x)    --j;
        if(i<=j)
        {
            double temp=Mass[i];
            Mass[i]=Mass[j];
            Mass[j]=temp;
            int temp2=Mass2[i];
            Mass2[i]=Mass2[j];
            Mass2[j]=temp2;
            i++;    j--;
        }
    } while(i<=j);
    if(low<j)   qSort2int(Mass,Mass2,low,j);
    if(i<high)  qSort2int(Mass,Mass2,i,high);
}

double GetOptimum(int func_num, double* xopt)
{
    FILE *fpt;
    char FileName[50];
    int res = 0;
    //CHANGE!
    sprintf(FileName, "input_data/shift_data_%d.txt",func_num);
    //sprintf(FileName, "input_data_22/shift_data_%d.txt",func_num);
    fpt = fopen(FileName,"r");
    if (fpt==NULL)
        printf("\n Error: Cannot open input file for reading 1 \n");
    for(int k=0;k<GNVars;k++)
        res = fscanf(fpt,"%lf",&xopt[k]);
    fclose(fpt);
    //CHANGE!
    cec17_test_func(xopt, tempF, GNVars, 1, func_num);
    //cec22_test_func(xopt, tempF, GNVars, 1, func_num);
    return tempF[0];
}
void SaveBestValues(int func_index)
{
    double temp = globalbest - fopt;
    if(temp <= 1E-8 && ResultsArray[ResTsize2-1] == MaxFEval)
        ResultsArray[ResTsize2-1] = NFEval;
    for(int stepFEcount=LastFEcount;stepFEcount<ResTsize2-1;stepFEcount++)
    {
        if(NFEval == stepsFEval[stepFEcount])
        {
            if(temp <= 1E-8)
                temp = 0;
            ResultsArray[stepFEcount] = temp;
            LastFEcount = stepFEcount;
        }
    }
}
class Optimizer
{
public:
    int MemorySize;
    int MemoryIter;
    int SuccessFilled;
    int MemoryCurrentIndex;
    int MemoryCurrentIndex2;
    int NVars;			    // размерность пространства
    int NIndsCurrent;
    int NIndsFront;
    int NIndsFrontMax;
    int newNIndsFront;
    int PopulSize;
    int func_num;
    int func_index;
    int TheChosenOne;
    int Generation;
    int PFIndex;

    double bestfit;
    double SuccessRate;
    double F;       /*параметры*/
    double Cr;
    double Right;		    // верхняя граница
    double Left;		    // нижняя граница

    double** Popul;	        // массив для частиц
    double** PopulFront;
    double** PopulTemp;
    double* FitArr;		// значения функции пригодности
    double* FitArrCopy;
    double* FitArrFront;
    double* Trial;
    double* tempSuccessCr;
    double* tempSuccessF;
    double* MemoryCr;
    double* MemoryF;
    double* FitDelta;
    double* Weights;

    int* Indices;
    int* Indices2;

    double* ord_best_arch;
    double* ord_medium_arch;
    double* ord_worst_arch;
    double* ord_best_popul;
    double* ord_medium_popul;
    double* ord_worst_popul;
    double* FitMass;
    double EB_hybrid_rate;
    double* FitTemp;

    // =====================
    // Phase 2 diagnostics members
    // =====================
    // CHANGE: these fields only observe the run. They do not change RDEx logic.
    bool diagnosticsEnabled;
    std::ofstream* diagnosticsStream;
    int diagnosticsRunId;
    unsigned diagnosticsRunSeed;
    int genEBTrials;
    int genStdTrials;
    int genEBSuccess;
    int genStdSuccess;
    double genSumF;
    double genSumActualCr;
    double genEBImprovement;
    double genStdImprovement;

    // CHANGE (late_v1): diagnostics for late-phase EB-rate control.
    bool genLateControlActive;
    double genRawEBRate;
    double genAdjustedEBRate;

    void EB_order(int prand, int Rand1, int Rand2);
    double* EB_hybrid_flag;
    void UpdateEB_hybrid_param(double* EB_hybrid_flag, double* FitArrFront, vector<double> FitTemp);

    void Initialize(int _newNInds, int _newNVars, int _newfunc_num, int _newfunc_index);
    void ConfigureDiagnostics(bool enabled, std::ofstream* diagStream, int runId, unsigned runSeed);
    void ResetGenerationDiagnostics();
    void RecordTrialDiagnostics(bool usedEB, double usedF, double actualCr, double previousFit, double newFit);
    void WriteGenerationDiagnostics();
    double ComputeFrontDiversity();
    void Clean();
    void MainCycle();
    void FindNSaveBest(bool init, int IndIter);
    void UpdateMemoryCr();
    double MeanWL(double* Vector, double* TempWeights);
    void RemoveWorst(int NInds, int NewNInds);
};
double cec_24(double* HostVector,int func_num)
{
    //CHANGE!
    cec17_test_func(HostVector, tempF, GNVars, 1, func_num);
    //cec22_test_func(HostVector, tempF, GNVars, 1, func_num);	
    NFEval++;
    return tempF[0];
}
void Optimizer::Initialize(int _newNInds, int _newNVars, int _newfunc_num, int _newfunc_index)
{
    NVars = _newNVars;
    NIndsCurrent = _newNInds;
    NIndsFront = _newNInds;
    NIndsFrontMax = _newNInds;
    PopulSize = _newNInds*2;
    Left = -100;
    Right = 100;
    Generation = 0;
    TheChosenOne = 0;
    MemorySize = 5;
    MemoryIter = 0;
    SuccessFilled = 0;
    SuccessRate = 0.5;
    func_num = _newfunc_num;
    func_index = _newfunc_index;
    for(int steps_k=0;steps_k!=ResTsize2-1;steps_k++)
        stepsFEval[steps_k] = 10000.0/double(ResTsize2-1)*GNVars*(steps_k+1);

    Popul = new double*[PopulSize];
    for(int i=0;i!=PopulSize;i++)
        Popul[i] = new double[NVars];
    PopulFront = new double*[NIndsFront];
    for(int i=0;i!=NIndsFront;i++)
        PopulFront[i] = new double[NVars];
    PopulTemp = new double*[PopulSize];
    for(int i=0;i!=PopulSize;i++)
        PopulTemp[i] = new double[NVars];
    FitArr = new double[PopulSize];
    FitArrCopy = new double[PopulSize];
    FitArrFront = new double[NIndsFront];
    Weights = new double[PopulSize];
    tempSuccessCr = new double[PopulSize];
    tempSuccessF = new double[PopulSize];
    FitDelta = new double[PopulSize];
    MemoryCr = new double[MemorySize];
    MemoryF = new double[MemorySize];
    Trial = new double[NVars];
    Indices = new int[PopulSize];
    Indices2 = new int[PopulSize];

	for (int i = 0; i<PopulSize; i++)
		for (int j = 0; j<NVars; j++)
			Popul[i][j] = Random(Left,Right);
    for(int i=0;i!=PopulSize;i++)
        tempSuccessCr[i] = 0;
    for(int i=0;i!=PopulSize;i++)
        tempSuccessF[i] = 0;
    for(int i=0;i!=MemorySize;i++)
        MemoryCr[i] = 1.0;
    for(int i=0;i!=MemorySize;i++)
        MemoryF[i] = 1.0;

    EB_hybrid_flag = new double[NIndsFrontMax];
    FitMass = new double[NIndsFrontMax];
    FitTemp = new double[NIndsFrontMax];
    EB_hybrid_rate = EB_hybrid_rate_init;

    // CHANGE: diagnostics are configured externally per run.
    diagnosticsEnabled = false;
    diagnosticsStream = nullptr;
    diagnosticsRunId = -1;
    diagnosticsRunSeed = 0u;
    ResetGenerationDiagnostics();
}
void Optimizer::ConfigureDiagnostics(bool enabled, std::ofstream* diagStream, int runId, unsigned runSeed)
{
    diagnosticsEnabled = enabled && (diagStream != nullptr);
    diagnosticsStream = diagStream;
    diagnosticsRunId = runId;
    diagnosticsRunSeed = runSeed;

    if(diagnosticsEnabled && diagnosticsStream && diagnosticsStream->tellp() == std::streampos(0))
    {
        (*diagnosticsStream)
            << "Generation,NFEval,FrontSize,SuccessRate,EBRate,AvgF,AvgActualCr,"
            << "EBTrials,StdTrials,EBSuccess,StdSuccess,EBImprove,StdImprove,"
            << "FrontDiversity,BestError,LateControlActive,RawEBRate,AdjustedEBRate,"
            << "RunId,RunSeed\n";
    }
}

void Optimizer::ResetGenerationDiagnostics()
{
    genEBTrials = 0;
    genStdTrials = 0;
    genEBSuccess = 0;
    genStdSuccess = 0;
    genSumF = 0.0;
    genSumActualCr = 0.0;
    genEBImprovement = 0.0;
    genStdImprovement = 0.0;
    genLateControlActive = false;
    genRawEBRate = EB_hybrid_rate;
    genAdjustedEBRate = EB_hybrid_rate;
}

void Optimizer::RecordTrialDiagnostics(bool usedEB, double usedF, double actualCr, double previousFit, double newFit)
{
    genSumF += usedF;
    genSumActualCr += actualCr;

    if(usedEB)
        genEBTrials++;
    else
        genStdTrials++;

    if(newFit <= previousFit)
    {
        const double improvement = previousFit - newFit;
        if(usedEB)
        {
            genEBSuccess++;
            genEBImprovement += improvement;
        }
        else
        {
            genStdSuccess++;
            genStdImprovement += improvement;
        }
    }
}

double Optimizer::ComputeFrontDiversity()
{
    if(NIndsFront <= 1)
        return 0.0;

    double avgVar = 0.0;
    for(int j = 0; j < NVars; ++j)
    {
        double mean = 0.0;
        for(int i = 0; i < NIndsFront; ++i)
            mean += PopulFront[i][j];
        mean /= static_cast<double>(NIndsFront);

        double var = 0.0;
        for(int i = 0; i < NIndsFront; ++i)
        {
            const double d = PopulFront[i][j] - mean;
            var += d * d;
        }
        var /= static_cast<double>(NIndsFront);
        avgVar += var;
    }
    avgVar /= static_cast<double>(NVars);
    return std::sqrt(avgVar);
}

void Optimizer::WriteGenerationDiagnostics()
{
    if(!diagnosticsEnabled || diagnosticsStream == nullptr)
        return;

    const int totalTrials = genEBTrials + genStdTrials;
    const double avgF = (totalTrials > 0) ? (genSumF / static_cast<double>(totalTrials)) : 0.0;
    const double avgActualCr = (totalTrials > 0) ? (genSumActualCr / static_cast<double>(totalTrials)) : 0.0;
    double bestError = globalbest - fopt;
    if(bestError <= 1e-8)
        bestError = 0.0;

    (*diagnosticsStream)
        << Generation << ","
        << NFEval << ","
        << NIndsFront << ","
        << SuccessRate << ","
        << EB_hybrid_rate << ","
        << avgF << ","
        << avgActualCr << ","
        << genEBTrials << ","
        << genStdTrials << ","
        << genEBSuccess << ","
        << genStdSuccess << ","
        << genEBImprovement << ","
        << genStdImprovement << ","
        << ComputeFrontDiversity() << ","
        << bestError << ","
        << (genLateControlActive ? 1 : 0) << ","
        << genRawEBRate << ","
        << genAdjustedEBRate << ","
        << diagnosticsRunId << ","
        << diagnosticsRunSeed << "\n";
}

void Optimizer::UpdateMemoryCr()
{
    if(SuccessFilled != 0)
    {
        MemoryCr[MemoryIter] = 0.5*(MeanWL(tempSuccessCr,FitDelta) + MemoryCr[MemoryIter]);
        MemoryF[MemoryIter] = MeanWL(tempSuccessF, FitDelta);
        MemoryIter = (MemoryIter+1)%MemorySize;
    }
}
double Optimizer::MeanWL(double* Vector, double* TempWeights)
{
    double SumWeight = 0;
    double SumSquare = 0;
    double Sum = 0;
    for(int i=0;i!=SuccessFilled;i++)
        SumWeight += TempWeights[i];
    for(int i=0;i!=SuccessFilled;i++)
        Weights[i] = TempWeights[i]/SumWeight;
    for(int i=0;i!=SuccessFilled;i++)
        SumSquare += Weights[i]*Vector[i]*Vector[i];
    for(int i=0;i!=SuccessFilled;i++)
        Sum += Weights[i]*Vector[i];
    if(fabs(Sum) > 1e-8)
        return SumSquare/Sum;
    else
        return 1.0;
}
void Optimizer::FindNSaveBest(bool init, int IndIter)
{
    if(FitArr[IndIter] <= bestfit || init)
        bestfit = FitArr[IndIter];
    if(bestfit < globalbest || init)
	{
		globalbest = bestfit;		
	}
}
void Optimizer::RemoveWorst(int _NIndsFront, int _newNIndsFront)
{
    int PointsToRemove = _NIndsFront - _newNIndsFront;
    for(int L=0;L!=PointsToRemove;L++)
    {
        double WorstFit = FitArrFront[0];
        int WorstNum = 0;
        for(int i=1;i!=_NIndsFront;i++)
        {
            if(FitArrFront[i] > WorstFit)
            {
                WorstFit = FitArrFront[i];
                WorstNum = i;
            }
        }
        for(int i=WorstNum;i!=_NIndsFront-1;i++)
        {
            for(int j=0;j!=NVars;j++)
                PopulFront[i][j] = PopulFront[i+1][j];
            FitArrFront[i] = FitArrFront[i+1];
            FitMass[i] = FitMass[i+1];
        }
    }
}
void Optimizer::EB_order(int prand, int Rand1, int Rand2) {//double* pos1, double* pos2, double* pos3, double* pos4, double* pos1_Fit, double* pos2_Fit, double* pos3_Fit, double* pos4_Fit
    double* pos1 = Popul[prand];
    double pos1Fit = FitArr[prand];
    double* pos3 = PopulFront[Rand1];
    double pos3Fit = FitArrFront[Rand1];
    double* pos4_arch = Popul[Rand2];
    double pos4Fit_arch = FitArr[Rand2];
    if (pos1Fit <= pos3Fit && pos1Fit <= pos4Fit_arch){
        ord_best_arch = pos1;
        if (pos3Fit <= pos4Fit_arch){
            ord_medium_arch = pos3;
            ord_worst_arch = pos4_arch;
        }
        else{
            ord_medium_arch = pos4_arch;
            ord_worst_arch = pos3;
        }
    }
    else if (pos3Fit <= pos1Fit && pos3Fit <= pos4Fit_arch){
        ord_best_arch = pos3;
        if (pos1Fit <= pos4Fit_arch){
            ord_medium_arch = pos1;
            ord_worst_arch = pos4_arch;
        }
        else{
            ord_medium_arch = pos4_arch;
            ord_worst_arch = pos1;
        }
    }
    else if (pos4Fit_arch <= pos1Fit && pos4Fit_arch <= pos3Fit){
        ord_best_arch = pos4_arch;
        if (pos1Fit <= pos3Fit){
            ord_medium_arch = pos1;
            ord_worst_arch = pos3;
        }
        else{
            ord_medium_arch = pos3;
            ord_worst_arch = pos1;
        }
    }
    double* pos4_popul = Popul[Rand2];
    double pos4Fit_popul = FitArr[Rand2];
    if (pos1Fit <= pos3Fit && pos1Fit <= pos4Fit_popul){
        ord_best_popul = pos1;
        if (pos3Fit <= pos4Fit_popul){
            ord_medium_popul = pos3;
            ord_worst_popul = pos4_popul;
        }
        else{
            ord_medium_popul = pos4_popul;
            ord_worst_popul = pos3;
        }
    }
    else if (pos3Fit <= pos1Fit && pos3Fit <= pos4Fit_popul){
        ord_best_popul = pos3;
        if (pos1Fit <= pos4Fit_popul){
            ord_medium_popul = pos1;
            ord_worst_popul = pos4_popul;
        }
        else{
            ord_medium_popul = pos4_popul;
            ord_worst_popul = pos1;
        }
    }
    else if (pos4Fit_popul <= pos1Fit && pos4Fit_popul <= pos3Fit){
        ord_best_popul = pos4_popul;
        if (pos1Fit <= pos3Fit){
            ord_medium_popul = pos1;
            ord_worst_popul = pos3;
        }
        else{
            ord_medium_popul = pos3;
            ord_worst_popul = pos1;
        }
    }
}
void Optimizer::UpdateEB_hybrid_param(double* EB_hybrid_flag, double* FitArrFront, vector<double> FitTemp) {
    double SumEB_DeltaFit = 0;
    double SumOrigin_DeltaFit = 0;
    for (int ChosenOne = 0; ChosenOne != NIndsFront; ChosenOne++) {
        if (EB_hybrid_flag[ChosenOne] == 1) {
            if (FitTemp[ChosenOne] <= FitArrFront[ChosenOne]) {
                SumEB_DeltaFit += FitArrFront[ChosenOne] - FitTemp[ChosenOne];
            }
        }
        else{
            if (FitTemp[ChosenOne] <= FitArrFront[ChosenOne]) {
                SumOrigin_DeltaFit += FitArrFront[ChosenOne] - FitTemp[ChosenOne];
            }
        }
    }

    // CHANGE (late_smooth_only_v1): keep baseline behavior by default, then
    // optionally apply a *late-phase-only* smoothing-only rule after the
    // normal EB stage begins. Pre-70% FE behavior is unchanged.
    double newEBRate = EB_hybrid_rate_init;
    if (SumEB_DeltaFit != 0 && SumOrigin_DeltaFit != 0){
        newEBRate = SumEB_DeltaFit / (SumEB_DeltaFit + SumOrigin_DeltaFit);
        double EB_limit_min = 0;
        double EB_limit_max = 1;
        if (newEBRate > EB_limit_max){
            newEBRate = EB_limit_max;
        }
        else if (newEBRate < EB_limit_min){
            newEBRate = EB_limit_min;
        }
    }

    genLateControlActive = false;
    genRawEBRate = newEBRate;

    const double progress = (MaxFEval > 0) ? (static_cast<double>(NFEval) / static_cast<double>(MaxFEval)) : 0.0;
    if (ENABLE_LATE_EB_CONTROL && progress >= LATE_EB_ENABLE_PROGRESS_FRACTION && SumEB_DeltaFit != 0 && SumOrigin_DeltaFit != 0){
        genLateControlActive = true;
        const double previousEBRate = EB_hybrid_rate;
        newEBRate = LATE_EB_SMOOTH_OLD_WEIGHT * previousEBRate
                  + (1.0 - LATE_EB_SMOOTH_OLD_WEIGHT) * newEBRate;

    }
    else if (!(SumEB_DeltaFit != 0 && SumOrigin_DeltaFit != 0)){
        // Keep the original fallback when one branch contributed no successful
        // improvement in the current generation.
        newEBRate = EB_hybrid_rate_init;
    }

    EB_hybrid_rate = newEBRate;
    genAdjustedEBRate = EB_hybrid_rate;
}
void Optimizer::MainCycle()
{
    vector<double> FitTemp2;
    vector<double> FitTemp_prand;
    for(int IndIter=0;IndIter<NIndsFront;IndIter++)
    {
        FitArr[IndIter] = cec_24(Popul[IndIter],func_num);
        FindNSaveBest(IndIter == 0,IndIter);
        if(!globalbestinit || bestfit < globalbest)
        {
            globalbest = bestfit;
            globalbestinit = true;
        }
        SaveBestValues(func_index);
    }
    double minfit = FitArr[0];
    double maxfit = FitArr[0];
    for(int i=0;i!=NIndsFront;i++)
    {
        FitArrCopy[i] = FitArr[i];
        Indices[i] = i;
        maxfit = max(maxfit,FitArr[i]);
        minfit = min(minfit,FitArr[i]);
    }
    if(minfit != maxfit)
        qSort2int(FitArrCopy,Indices,0,NIndsFront-1);
    for(int i=0;i!=NIndsFront;i++)
    {
        for(int j=0;j!=NVars;j++)
            PopulFront[i][j] = Popul[Indices[i]][j];
        FitArrFront[i] = FitArrCopy[i];
        FitMass[i] = FitArrFront[i];
    }
    PFIndex = 0;
    while(NFEval < MaxFEval)
    {
        // CHANGE: reset generation diagnostics before collecting new stats.
        ResetGenerationDiagnostics();

        double meanF = 0.4+tanh(SuccessRate*5)*0.25;
        double sigmaF = STANDARD_BRANCH_SIGMA_F; // CHANGE (lsv1_start75_sigmaF025_xi072)
        minfit = FitArr[0];
        maxfit = FitArr[0];
        for(int i=0;i!=NIndsFront;i++)
        {
            FitArrCopy[i] = FitArr[i];
            Indices[i] = i;
            maxfit = max(maxfit,FitArr[i]);
            minfit = min(minfit,FitArr[i]);
        }
        if(minfit != maxfit)
            qSort2int(FitArrCopy,Indices,0,NIndsFront-1);
        minfit = FitArrFront[0];
        maxfit = FitArrFront[0];
        for(int i=0;i!=NIndsFront;i++)
        {
            FitArrCopy[i] = FitArrFront[i];
            Indices2[i] = i;
            maxfit = max(maxfit,FitArrFront[i]);
            minfit = min(minfit,FitArrFront[i]);
        }
        if(minfit != maxfit)
            qSort2int(FitArrCopy,Indices2,0,NIndsFront-1);
        FitTemp2.resize(NIndsFront);
        for(int i=0;i!=NIndsFront;i++)
            FitTemp2[i] = exp(-double(i)/double(NIndsFront)*3);     
        std::discrete_distribution<int> ComponentSelectorFront (FitTemp2.begin(),FitTemp2.end());
        
        int prand = 0;
        int Rand1 = 0;
        int Rand2 = 0;
        int psizeval = max(2,int(NIndsFront*PSIZEVAL_XI_COEFF*exp(-SuccessRate*PSIZEVAL_k_COEFF)));// CHANGE (lsv1_start75_sigmaF025_xi072)
        FitTemp_prand.resize(NIndsFront);
        for (int i = 0; i != NIndsFront; i++)
            FitTemp_prand[i] = 3.0 * (NIndsFront - i);
        int psizeval2 = NIndsFront * EB_PSIZEVAL2_COEFF * (1 - 0.5 * (double)NFEval / (double)MaxFEval);
        if (psizeval2 <= 1)
            psizeval2 = 2;
        std::discrete_distribution<int> ComponentSelectorFront2 (FitTemp_prand.begin(), FitTemp_prand.begin() + psizeval2);
        std::discrete_distribution<int> ComponentSelectorFront3 (FitTemp_prand.begin(),FitTemp_prand.end());

        for(int IndIter=0;IndIter<NIndsFront;IndIter++)
        {
            TheChosenOne = IntRandom(NIndsFront);
            MemoryCurrentIndex = IntRandom(MemorySize);
            MemoryCurrentIndex2 = IntRandom(MemorySize+1);
            do
                prand = Indices[IntRandom(psizeval)];
            while(prand == TheChosenOne);
            do
                Rand1 = Indices2[ComponentSelectorFront(generator_uni_i_3)];
            while(Rand1 == prand);
            do
                Rand2 = Indices[IntRandom(NIndsFront)];
            while(Rand2 == prand || Rand2 == Rand1);
            do
                F = NormRand(meanF,sigmaF);
            while(F < 0.0 || F > 1.0);
            Cr = NormRand(MemoryCr[MemoryCurrentIndex],0.05);
            Cr = min(max(Cr,0.0),1.0);
            double ActualCr = 0;
            int WillCrossover = IntRandom(NVars);
            const double PreviousFrontFit = FitArrFront[TheChosenOne];
            bool usedEBThisTrial = false;

            double Rand_EB = Random(0, 1);
            if ((double)NFEval / (double)MaxFEval < 0.7){
                Rand_EB = 2;
            }
            if ((Rand_EB * (1 - NFEval / MaxFEval)) < EB_hybrid_rate){
                usedEBThisTrial = true;
                EB_hybrid_flag[TheChosenOne] = 1;
 
                do
                    prand = Indices[ComponentSelectorFront2(generator_uni_i_3)];
                while(prand == TheChosenOne);
                do
                    Rand1 = Indices2[ComponentSelectorFront3(generator_uni_i_3)];
                while (Rand1 == prand);
                do
                    Rand2 = Indices[ComponentSelectorFront3(generator_uni_i_3)];
                while (Rand2 == prand || Rand2 == Rand1);
                EB_order(prand, Rand1, Rand2);
                do {
                    if (MemoryCurrentIndex2 < MemorySize)
                        F = CachyRand(MemoryF[MemoryCurrentIndex2], 0.1);
                    else
                        F = CachyRand(0.9, 0.1);
                } while (F < 0.0);
                if (F > 1.0)
                    F = 1.0;

		        if ((double)NFEval / (double)MaxFEval < 0.6 && F > 0.7)
                    F = 0.7;

                if (MemoryCurrentIndex2 < MemorySize) {
                    if (MemoryCr[MemoryCurrentIndex2] < 0)
                        Cr = 0;
                    else
                        Cr = NormRand(MemoryCr[MemoryCurrentIndex2], 0.1);
                } else
                    Cr = NormRand(0.9, 0.1);
                if (Cr >= 1)
                    Cr = 1;
                if (Cr <= 0)
                    Cr = 0;

                if ((double)NFEval / (double)MaxFEval < 0.25)
                    Cr = max(Cr, 0.7);
                if ((double)NFEval / (double)MaxFEval < 0.5)
                    Cr = max(Cr, 0.6);

                bool perturbation = rand() / (double)RAND_MAX < 0.4;
                                
                for(int j=0;j!=NVars;j++)
                {
                    if(Random(0,1) < Cr || WillCrossover == j)
                    {
                        Trial[j] = PopulFront[TheChosenOne][j] +
                        F * (ord_best_arch[j] - PopulFront[TheChosenOne][j]) +
                        F * (ord_medium_arch[j] - ord_worst_arch[j]);                                                 
                        if(Trial[j] < Left)
                            Trial[j] = Random(Left,Right);
                        if(Trial[j] > Right)
                            Trial[j] = Random(Left,Right);
                        ActualCr++;
                    }
                    else
                        Trial[j] = perturbation ? CachyRand(PopulFront[TheChosenOne][j], 0.1) : PopulFront[TheChosenOne][j];
                }
            }
            else{
                EB_hybrid_flag[TheChosenOne] = 0;
                bool perturbation = rand() / (double)RAND_MAX < 0.4;
                for(int j=0;j!=NVars;j++)
                {
                    if(Random(0,1) < Cr || WillCrossover == j)
                    {
                        Trial[j] = PopulFront[TheChosenOne][j] + F*(Popul[prand][j] - PopulFront[TheChosenOne][j]) + F*(PopulFront[Rand1][j] - Popul[Rand2][j]);
                        if(Trial[j] < Left)
                            Trial[j] = Random(Left,Right);
                        if(Trial[j] > Right)
                            Trial[j] = Random(Left,Right);
                        ActualCr++;
                    }
                    else
                        Trial[j] = perturbation ? CachyRand(PopulFront[TheChosenOne][j], 0.1) : PopulFront[TheChosenOne][j];
                }
            }
            
            ActualCr = ActualCr / double(NVars);
            double TempFit = cec_24(Trial,func_num);
            FitTemp[TheChosenOne] = TempFit;
            RecordTrialDiagnostics(usedEBThisTrial, F, ActualCr, PreviousFrontFit, TempFit);
            if(TempFit <= FitArrFront[TheChosenOne])
            {
                for(int j=0;j!=NVars;j++)
                {
                    Popul[NIndsCurrent+SuccessFilled][j] = Trial[j];
                    PopulFront[PFIndex][j] = Trial[j];
                }
                FitArr[NIndsCurrent+SuccessFilled] = TempFit;
                FitArrFront[PFIndex] = TempFit;
                FindNSaveBest(false,NIndsCurrent+SuccessFilled);
                tempSuccessCr[SuccessFilled] = ActualCr;//Cr;
                tempSuccessF[SuccessFilled] = F;//Cr;
                FitDelta[SuccessFilled] = fabs(FitArrFront[TheChosenOne]-TempFit);
                SuccessFilled++;
                PFIndex = (PFIndex + 1)%NIndsFront;
            }
            SaveBestValues(func_index);
        }
        for (int ChosenOne = 0; ChosenOne != NIndsFront; ChosenOne++) {
            FitTemp_prand[ChosenOne] = FitTemp[ChosenOne];            
        }
        UpdateEB_hybrid_param(EB_hybrid_flag, FitMass, FitTemp_prand);
        for (int ChosenOne = 0; ChosenOne != NIndsFront; ChosenOne++) {
            FitMass[ChosenOne] = FitArrFront[ChosenOne];            
        }
        SuccessRate = double(SuccessFilled)/double(NIndsFront);

        // CHANGE: write one diagnostics row per generation before the front is shrunk.
        WriteGenerationDiagnostics();

        newNIndsFront = int(double(4-NIndsFrontMax)/double(MaxFEval)*NFEval + NIndsFrontMax);
        RemoveWorst(NIndsFront,newNIndsFront);
        NIndsFront = newNIndsFront;
        UpdateMemoryCr();
        NIndsCurrent = NIndsFront + SuccessFilled;
        SuccessFilled = 0;
        Generation++;		
        if(NIndsCurrent > NIndsFront)
        {
            minfit = FitArr[0];
            maxfit = FitArr[0];
            for(int i=0;i!=NIndsCurrent;i++)
            {
                Indices[i] = i;
                maxfit = max(maxfit,FitArr[i]);
                minfit = min(minfit,FitArr[i]);
            }
            if(minfit != maxfit)
                qSort2int(FitArr,Indices,0,NIndsCurrent-1);
            NIndsCurrent = NIndsFront;
            for(int i=0;i!=NIndsCurrent;i++)
                for(int j=0;j!=NVars;j++)
                    PopulTemp[i][j] = Popul[Indices[i]][j];
            for(int i=0;i!=NIndsCurrent;i++)
                for(int j=0;j!=NVars;j++)
                    Popul[i][j] = PopulTemp[i][j];
        }
    }
}
void Optimizer::Clean()
{
    // CHANGE: use delete[] for arrays allocated with new[]. This is an
    // infrastructure safety fix; it does not change RDEx search logic.
    delete[] Trial;
    for(int i=0;i!=PopulSize;i++)
        delete[] Popul[i];
    for(int i=0;i!=NIndsFrontMax;i++)
        delete[] PopulFront[i];
    for(int i=0;i!=PopulSize;i++)
        delete[] PopulTemp[i];
    delete[] PopulTemp;
    delete[] Popul;
    delete[] PopulFront;
    delete[] FitArr;
    delete[] FitArrCopy;
    delete[] FitArrFront;
    delete[] Indices;
    delete[] Indices2;
    delete[] tempSuccessCr;
    delete[] tempSuccessF;
    delete[] FitDelta;
    delete[] MemoryCr;
    delete[] MemoryF;
    delete[] Weights;
    delete[] FitMass;
    delete[] EB_hybrid_flag;
    delete[] FitTemp;
}

int main(int argc, char** argv)
{
    unsigned t0g = clock(), t1g;
    int TotalNRuns;
    if(Benchmark == 17)
        TotalNRuns = NUM_RUNS; // 25 for CEC 2024, 51 for CEC 2017
    else
        TotalNRuns = 30;

    // =====================
    // Phase 1: experiment harness setup
    // =====================
    // CHANGE: one dedicated folder per experiment/variant.
    const std::filesystem::path experimentDir = BuildExperimentDir();
    const std::filesystem::path checkpointsDir = experimentDir / "checkpoints";
    const std::filesystem::path diagnosticsDir = experimentDir / "diagnostics";
    std::filesystem::create_directories(experimentDir);
    std::filesystem::create_directories(checkpointsDir);
    std::filesystem::create_directories(diagnosticsDir);

    // CHANGE: one seed list reused by the baseline and every future candidate.
    const std::vector<unsigned> runSeeds = BuildRunSeeds(TotalNRuns);

    // CHANGE: save experiment metadata once so every batch is reproducible.
    {
        std::ofstream configFile((experimentDir / "experiment_config.txt").string());
        configFile << "Algorithm=" << algName << "\n";
        configFile << "Variant=" << variantName << "\n";
        configFile << "ExperimentTag=" << experimentTag << "\n";
        configFile << "Benchmark=" << Benchmark << "\n";
        configFile << "Runs=" << TotalNRuns << "\n";
        configFile << "UseFixedRunSeeds=" << (USE_FIXED_RUN_SEEDS ? 1 : 0) << "\n";
        configFile << "MasterSeed=" << MASTER_SEED << "\n";
        configFile << "SaveDiagnostics=" << (SAVE_DIAGNOSTICS ? 1 : 0) << "\n";
        configFile << "DiagnosticRunIndex=" << DIAGNOSTIC_RUN_INDEX << "\n";
        configFile << "EB_hybrid_rate_init=" << EB_hybrid_rate_init << "\n";
        configFile << "EnableLateEBControl=" << (ENABLE_LATE_EB_CONTROL ? 1 : 0) << "\n";
        configFile << "LateEBEnableProgressFraction=" << LATE_EB_ENABLE_PROGRESS_FRACTION << "\n";
        configFile << "LateEBSmoothOldWeight=" << LATE_EB_SMOOTH_OLD_WEIGHT << "\n";
        configFile << "StandardBranchSigmaF=" << STANDARD_BRANCH_SIGMA_F << "\n";
        configFile << "PsizevalXiCoeff=" << PSIZEVAL_XI_COEFF << "\n";
        configFile << "PsizevalkCoeff=" << PSIZEVAL_k_COEFF << "\n";
    }

    {
        std::ofstream seedsFile((experimentDir / "run_seeds.csv").string());
        seedsFile << "Run,Seed\n";
        for(int run = 0; run < TotalNRuns; ++run)
            seedsFile << run << "," << runSeeds[run] << "\n";
    }

    if(TimeComplexity) // Works for CEC 2017/2024
    {
        std::ofstream fout_t((experimentDir / "time_complexity.txt").string());
        cout << "Running time complexity code" << endl;
        double T1, T2;
        unsigned t1 = clock(), t0;
        GNVars = 30;
        MaxFEval = 10000;
        double* xtmp = new double[GNVars];
        for(int j = 0; j != GNVars; j++)
            xtmp[j] = 0;

        t0 = clock();
        for(int func_num = 1; func_num != 31; func_num++)
        {
            if(func_num == 2)
                continue;
            NFEval = 0;
            for(int j = 0; j != MaxFEval; j++)
            {
                // CHANGE !
                cec17_test_func(xtmp, tempF, GNVars, 1, func_num);
                //cec22_test_func(xtmp, tempF, GNVars, 1, func_num);
            }
        }
        t1 = clock() - t0;
        T1 = double(t1) / 29.0;
        cout << "T1 = " << T1 << endl;
        fout_t << "T1 = " << T1 << endl;

        t0 = clock();
        for(int func_num = 1; func_num != 31; func_num++)
        {
            if(func_num == 2)
                continue;
            SetGlobalSeed(MASTER_SEED + static_cast<unsigned>(func_num));
            globalbestinit = false;
            LastFEcount = 0;
            NFEval = 0;
            int PopSize = 20;
            fopt = 100 * func_num;
            ResetResultsArray();
            Optimizer OptZ;
            OptZ.Initialize(PopSize * GNVars, GNVars, 1, func_num);
            OptZ.MainCycle();
            OptZ.Clean();
        }
        t1 = clock() - t0;
        T2 = double(t1) / 29.0;
        cout << "T2 = " << T2 << endl;
        fout_t << "T2 = " << T2 << endl;

        delete[] xtmp;
    }

    // Should be 0:4 for CEC 2017, 0:1 for CEC 2022, 1:2 for CEC 2024
    for(int GNVarsIter = 1; GNVarsIter != 2; GNVarsIter++)
    {
        int maxNFunc;
        if(Benchmark == 17)
        {
            if(GNVarsIter == 0)
            {
                GNVars = 10;
                MaxFEval = 100000;
            }
            if(GNVarsIter == 1)
            {
                GNVars = 30;
                MaxFEval = 300000;
            }
            if(GNVarsIter == 2)
            {
                GNVars = 50;
                MaxFEval = 500000;
            }
            if(GNVarsIter == 3)
            {
                GNVars = 100;
                MaxFEval = 1000000;
            }
            maxNFunc = 30;
        }
        else
        {
            if(GNVarsIter == 0)
            {
                GNVars = 10;
                MaxFEval = 200000;
            }
            if(GNVarsIter == 1)
            {
                GNVars = 20;
                MaxFEval = 1000000;
            }
            maxNFunc = 12;
        }

        // CHANGE: save both per-function summaries and per-run final errors.
        std::string prefix = BuildExperimentPrefix() + "_D" + std::to_string(GNVars);
        std::ofstream summary_csv((experimentDir / (prefix + "_summary_stats.csv")).string());
        std::ofstream final_error_csv((experimentDir / (prefix + "_final_errors.csv")).string());
        summary_csv << "Function,Best,Worst,Mean,Std,Median\n";
        final_error_csv << "Dimension,Function,Run,Seed,FinalError\n";

        int func_index = 1;
        for(int func_num = 1; func_num < maxNFunc + 1; func_num++)
        {
            if(func_num == 2 && Benchmark == 17) // Skip F2
                continue;

            fopt = 100 * func_num;
            sprintf(buffer, "%s_%s_%s_D%d_F%d.txt", algName, variantName, experimentTag, GNVars, func_index);
            std::string outFile = (checkpointsDir / buffer).string();
            ofstream fout(outFile);

            int PopSize = 18;
            std::vector<double> finalErrors;
            finalErrors.reserve(TotalNRuns);

            for(int run = 0; run != TotalNRuns; run++)
            {
                cout << "func\t" << func_num << "\trun\t" << run << endl;

                // CHANGE: same seed index for baseline and candidate code.
                SetGlobalSeed(runSeeds[run]);

                // CHANGE: hard reset result storage before every run.
                ResetResultsArray();
                globalbestinit = false;
                LastFEcount = 0;
                NFEval = 0;

                std::ofstream diagFile;
                Optimizer OptZ;
                OptZ.Initialize(PopSize * GNVars, GNVars, func_num, func_index);

                // CHANGE: save generation-level diagnostics for selected run(s).
                bool enableDiagForRun = SAVE_DIAGNOSTICS && (run == DIAGNOSTIC_RUN_INDEX);
                if(enableDiagForRun)
                {
                    std::string diagName = prefix + "_F" + std::to_string(func_num) + "_run" + std::to_string(run) + "_diagnostics.csv";
                    diagFile.open((diagnosticsDir / diagName).string());
                    OptZ.ConfigureDiagnostics(true, &diagFile, run, runSeeds[run]);
                }
                else
                {
                    OptZ.ConfigureDiagnostics(false, nullptr, run, runSeeds[run]);
                }

                OptZ.MainCycle();
                OptZ.Clean();

                if(diagFile.is_open())
                    diagFile.close();

                for(int j = 0; j != ResTsize2; j++)
                    ResultsArray2[run][j] = ResultsArray[j];

                double finalError = ResultsArray[ResTsize2 - 2]; // last checkpoint
                if(std::isnan(finalError))
                {
                    finalError = globalbest - fopt;
                    if(finalError <= 1e-8)
                        finalError = 0.0;
                }

                finalErrors.push_back(finalError);
                final_error_csv
                    << GNVars << ","
                    << "F" << func_num << ","
                    << run << ","
                    << runSeeds[run] << ","
                    << finalError << "\n";
            }

            for(int i = 0; i != 1000; i++)
            {
                for(int j = 0; j != TotalNRuns; j++)
                    fout << ResultsArray2[j][i] << "\t";
                fout << "\n";
            }

            double bestVal = *std::min_element(finalErrors.begin(), finalErrors.end());
            double worstVal = *std::max_element(finalErrors.begin(), finalErrors.end());
            double meanVal = ComputeMean(finalErrors);
            double stdVal = ComputeStdSample(finalErrors, meanVal);
            double medianVal = ComputeMedian(finalErrors);

            summary_csv << "F" << func_num << ","
                        << bestVal << ","
                        << worstVal << ","
                        << meanVal << ","
                        << stdVal << ","
                        << medianVal << "\n";

            func_index++;
        }

        summary_csv.close();
        final_error_csv.close();
    }

    t1g = clock() - t0g;
    double T0g = t1g / double(CLOCKS_PER_SEC);
    cout << "Time spent: " << T0g << endl;
    return 0;
}
