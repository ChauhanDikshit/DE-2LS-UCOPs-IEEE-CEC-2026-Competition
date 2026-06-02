%% main_uscore_BCSOP_pairwise.m
%  U-score Calculation for CEC 2026 CSOPs
%  ------------------------------------------------------------------------
%  Author      : Dikshit Chauhan
%  Affiliation : Department of Electrical and Computer Engineering,
%                National University of Singapore
%  Contact     : dikshitchauhan608@gmail.com 
%  Purpose     : This script computes the pairwise U-score for unconstrained
%                single-objective optimization problems (UCSOPs) using the
%                official CEC 2026-style accuracy and speed scoring procedure.
% Required scoring function: uscore_BCSOPs.m
%
% Expected data for each algorithm/function:
%   num x trial matrix of Min_EV values, or
%   [FEs, Min_EV_run1, ..., Min_EV_run25].
%
%  Created/Updated: 02-Jun-2026
% ========================================================================

clear all;
clc;
close all;

%% ===================== USER SETTINGS =====================

pro   = 29;      % number of UCSOP functions
trial = 25;      % number of independent runs
num   = 1000;    % number of saved progress points
D     = 30;      % dimension, used only in filenames
epsEV  = 1e-8;      % precision cutoff/tie tolerance for Min_EV
saveResults = true;

scriptDir = fileparts(mfilename('fullpath'));
if isempty(scriptDir)
    scriptDir = pwd;
end
addpath(scriptDir);

if exist('uscore_BCSOPs', 'file') ~= 2
    error(['uscore_nAlgorithms_UCSOP_pairwise.m was not found. ', ...
           'Place it in the same folder as this script or add it to the MATLAB path.']);
end

% -------------------------------------------------------------------------
% Select algorithms here.
% Each algFns entry must match the corresponding algNames entry.
algNames = {'LSRTDE','mLSHADE_LR','jSO','BlockEA','IEACOP','RDEx','DE-2LS'};
algFns   = {@load_LSRTDE,@load_mLSHADE,@load_jSO,@load_BlockEA,@load_IEACOP,@load_RDEx,@load_DE_2LS};
% -------------------------------------------------------------------------
% Build algs struct used by the scorer.
% -------------------------------------------------------------------------
algs = struct('name', {}, 'getT', {});
for a = 1:numel(algNames)
    algs(a).name = algNames{a};

    switch algNames{a}
        case {'jSO','BlockEA','IEACOP','mLSHADE_LR','LSRTDE','RDEx','DE-2LS'}
            algs(a).getT = @(f) algFns{a}(scriptDir, f, num, trial);
        otherwise
            error('Unknown algorithm name: %s', algNames{a});
    end
end

nAlg = numel(algs);

%% ===================== PAIRWISE BOUND-CONSTRAINED SOP U-SCORE =====================

[SR, score, scoreTbl, rankTbl, srTotAll, Data, speedPart, accuracyPart] = uscore_BCSOPs(pro, trial, num, algs, epsEV);
nAlg = numel(algs);
% ---------------- Overall U-score ----------------
u_score = sum(score, 1);

fprintf('\nU-score of all algorithms using pairwise UCSOP code:\n');
disp(array2table(u_score, 'VariableNames', local_valid_names(algNames, 'Uscore_')));

fprintf('Best U-score: %.6g\n', max(u_score));

%% ===================== DETAILED OUTPUT TABLE =====================
algLabels = string({algs.name});
algLabels = string(local_valid_names(algNames, ''));

accuracyPart = accuracyPart(:, 1:nAlg);
speedPart    = speedPart(:, 1:nAlg);
rawTotal     = accuracyPart + speedPart;
adjustedU    = score(:, 1:nAlg);
rankPart     = SR(:, 1:nAlg);

allData = [accuracyPart, speedPart, rawTotal, adjustedU, rankPart];


% Create variable names automatically
varNames = [ ...
    strcat("Accuracy ", algLabels), ...
    strcat("Speed ", algLabels), ...
    strcat("RawTotal ", algLabels), ...
    strcat("Uscore ", algLabels), ...
    strcat("Rank ", algLabels) ...
];

% Convert to table
dataTbl = array2table(allData, 'VariableNames', cellstr(varNames));
% Add function number
dataTbl = addvars(dataTbl, (1:pro)', 'Before', 1, 'NewVariableNames', 'Function');

% ---------------- Add summary row ----------------
summaryRow = dataTbl(1, :);
for j = 1:width(summaryRow)
    if isnumeric(summaryRow{1,j})
        summaryRow{1,j} = NaN;
    end
end
summaryRow.Function = NaN;
% Sum accuracy, speed, raw total, adjusted U-score, and ranks
summaryRow{1, 2:end} = sum(dataTbl{:, 2:end}, 1);
dataTbl = [dataTbl; summaryRow];

% Overall table.
overallTbl = table(string(algNames(:)), accuracyPart(end,:)'*0, ...
    'VariableNames', {'Algorithm','Dummy'});
overallTbl.Dummy = [];
overallTbl.TotalAccuracy = sum(accuracyPart, 1)';
overallTbl.TotalSpeed    = sum(speedPart, 1)';
overallTbl.TotalUscore   = u_score(:);
overallTbl.RankSum       = sum(SR, 1)';

[~, bestIdx] = max(overallTbl.TotalUscore);
overallTbl.IsBest = false(height(overallTbl),1);
overallTbl.IsBest(bestIdx) = true;

disp('Overall summary:');
disp(overallTbl);

%% ===================== SAVE RESULTS =====================
if saveResults
outputDir = fullfile(pwd, 'USCORE_Output');
    if ~exist(outputDir, 'dir')
        mkdir(outputDir);
    end

    xlsxFile = fullfile(outputDir, sprintf('USCORE_UCSOP_pairwise_%dAlg_%dF_%dRuns.xlsx', nAlg, pro, trial));

writetable(overallTbl, xlsxFile, 'Sheet', 'Overall_Uscore');
writetable(dataTbl,    xlsxFile, 'Sheet', 'Accuracy_Speed_Uscore');
writetable(scoreTbl,   xlsxFile, 'Sheet', 'Score_Table');
writetable(rankTbl,    xlsxFile, 'Sheet', 'Rank_Table');

 % Optional diagnostic fields from Data, if available.
    if isstruct(Data)
        try
            diagTbl = struct2table(Data);
            writetable(diagTbl, xlsxFile, 'Sheet', 'Diagnostics');
        catch
            % Some Data structures may contain non-table-compatible fields.
            save(fullfile(outputDir, 'USCORE_Diagnostics_Data.mat'), 'Data');
        end
    end

    save(fullfile(outputDir, 'USCORE_UCSOP_workspace.mat'), ...
        'SR', 'score', 'scoreTbl', 'rankTbl', 'srTotAll', 'Data', ...
        'speedPart', 'accuracyPart', 'u_score', 'overallTbl', 'dataTbl', ...
        'algNames', 'pro', 'trial', 'num', 'D');

    fprintf('\nU-score Excel file saved:\n%s\n', xlsxFile);
    fprintf('Workspace MAT file saved in:\n%s\n', outputDir);
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% ===================== LOADERS =====================
% All loaders return a num x trial matrix of Min_EV values.
% The scoring function also supports a first FE column, but these loaders
% usually remove/ignore it and return only Min_EV curves.

function T = load_BlockEA(folderr, f, num, trial)
    file = fullfile(folderr, 'BlockEA', sprintf('BlockEA_F%d_Min_EV.mat', f));
    S = load(file);
    T0 = S.combinedMatrix;

    % Original code used rows 2:num+1. If the file has enough rows, keep that.
    % Otherwise use the first num rows.
    if size(T0,1) >= num + 1
        T = T0(2:num+1, :);
    else
        T = T0(1:num, :);
    end
    T = T(:, 1:trial);
end

function T = load_IEACOP(folderr, f, num, trial)
    file = fullfile(folderr, 'IEACOP', sprintf('#1570992402_F%d_Min_EV.mat', f));
    S = load(file);
    T = S.Min_EV(1:num, 1:trial);
end

function T = load_DE_2LS(folderr, f, num, trial)
    file = fullfile(folderr, 'DE-2LS_results', sprintf('DE-2LS_D30_F%d.txt', f));
    T = load(file);
    T = T(1:num, 1:trial);
end

function T = load_RDEx(folderr, f, num, trial)
    file = fullfile(folderr, 'RDEx',sprintf('RDEx_D30_F%d.txt', f));
    T = load(file);        % your original transpose
    T = T(1:num,:);
end

function T = load_mLSHADE(folder1, f, num, trial)
    % The original mLSHADE files have a function-index offset except F1.
    if f == 1
        ff = 1;
    else
        ff = f + 1;
    end
    file = fullfile(folder1, 'mLSHADE_LR', sprintf('mLSAHDE_LR_F#%d_D#30.mat', ff));
    S = load(file, '-ascii');
    T = S(1:num, 1:trial);
end

function T = load_LSRTDE(folderr, f, num, trial)
    file = fullfile(folderr, 'L-SRTDE', sprintf('L-SRTDE_F%d_D30.txt', f));
    T = load(file); T=T';
    T = T(1:num, 1:trial);
end

function T = load_jSO(folderr, f, num, trial)
    % The original jSO files have a function-index offset except F1.
    if f == 1
        ff = 1;
    else
        ff = f + 1;
    end
    file = fullfile(folderr, 'jSOa', sprintf('jSOa_D30_S%d.txt', ff));
    T = load(file);
    T = T(1:num, 1:trial);
end

%% ===================== TABLE HELPER =====================

function names = local_valid_names(rawNames, prefix)
    rawNames = string(rawNames);
    names = matlab.lang.makeValidName(strcat(prefix, rawNames));
    names = matlab.lang.makeUniqueStrings(cellstr(names));
end