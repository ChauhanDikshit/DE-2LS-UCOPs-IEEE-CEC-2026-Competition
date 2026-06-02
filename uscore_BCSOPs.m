function [SR, score, scoreTbl, rankTbl, srTotAll, Data, sr1, ac1] = ...
    uscore_BCSOPs(pro, trial, num, algs, epsEV)
%USCORE_NALGORITHMS_BCSOP_PAIRWISE
% Pairwise U-score for CEC 2026 bound-constrained single-objective problems.
%
% This function is for unconstrained / bound-constrained SOPs where each run
% stores only Min_EV (best-so-far error value). It follows the procedure in
% the CEC 2026 bound-constrained SOP/MOP technical report:
%
%   1) Accuracy score A:
%      Every trial is compared with every other trial using final Min_EV.
%      The smaller final Min_EV gets 1 point; equal values get 0.5 each.
%
%   2) Speed score S:
%      For each pair of trials i and j, define the pairwise reference level
%      as theta = max(finalEV_i, finalEV_j), i.e., the worse of the two final
%      errors. The faster trial is the one that first reached theta. It gets
%      1 point; equal first-reaching times get 0.5 each.
%
%   3) Problem score:
%      score = A + S. No global TGT and no correction factor cf are used.
%
% INPUTS
%   pro   : number of functions/problems, e.g., 29 for CEC2017 SOPs
%   trial : number of independent runs per algorithm, e.g., 25
%   num   : number of saved sampling points, e.g., 1001 for MaxFEs=10000D
%   algs  : struct array with fields:
%           algs(a).name : algorithm name
%           algs(a).getT : function handle, X = algs(a).getT(f)
%   epsEV : precision/tie tolerance for Min_EV. Default = 1e-8.
%
% ACCEPTED OUTPUT FORMAT FROM algs(a).getT(f)
%   Option 1: numeric matrix without FE column
%       X = num x trial matrix of Min_EV values.
%
%   Option 2: numeric matrix with FE column
%       X(:,1) = FEs
%       X(:,2:end) = Min_EV runs 1..trial
%
%   Option 3: struct output
%       X.Min_EV or X.MinEV or X.EV or X.Obj or X.Fitness = num x trial
%       X.FEs or X.FE is optional.
%
% OUTPUTS
%   SR        : pro x nAlg rank matrix. Rank 1 is best. Larger score is better.
%   score     : pro x nAlg official problem score = accuracy + speed.
%   scoreTbl  : table of scores with a final sum row.
%   rankTbl   : table of ranks with a final sum row.
%   srTotAll  : same as score, kept for compatibility with previous code.
%   Data      : diagnostic struct.
%   sr1       : pro x nAlg speed score S.
%   ac1       : pro x nAlg accuracy score A.
%
% IMPORTANT
%   - The result curves must be best-so-far Min_EV curves.
%   - Values smaller than epsEV are set to zero, following the precision cutoff.
%   - Larger U-score is better.

    if nargin < 5 || isempty(epsEV)
        epsEV = 1e-8;
    end

    if ~isstruct(algs) || ~isfield(algs, 'name') || ~isfield(algs, 'getT')
        error('algs must be a struct array with fields name and getT.');
    end

    nAlg = numel(algs);
    nTotalTrials = nAlg * trial;
    nPairs = nTotalTrials * (nTotalTrials - 1) / 2;

    % Output arrays
    ac1      = zeros(pro, nAlg);    % accuracy score A
    sr1      = zeros(pro, nAlg);    % speed score S
    score    = zeros(pro, nAlg);    % final score A + S
    srTotAll = zeros(pro, nAlg);    % compatibility name
    SR       = zeros(pro, nAlg);    % rank per function

    % Diagnostics
    Data = struct();
    Data.epsEV = epsEV;
    Data.nAlgorithms = nAlg;
    Data.nTrialsPerAlgorithm = trial;
    Data.nTotalTrials = nTotalTrials;
    Data.nPairsPerFunction = nPairs;
    Data.MeanFinalEV = nan(pro, nAlg);
    Data.BestFinalEV = nan(pro, nAlg);
    Data.WorstFinalEV = nan(pro, nAlg);
    Data.NumFiniteFinalEV = zeros(pro, nAlg);
    Data.PairwiseAccuracyTrialScore = cell(pro, 1);
    Data.PairwiseSpeedTrialScore = cell(pro, 1);

    algLabels = matlab.lang.makeValidName(string({algs.name}));
    algLabels = matlab.lang.makeUniqueStrings(cellstr(algLabels));
    tableVarNames = [{'Function'}, algLabels];

    for f = 1:pro

        % ==============================================================
        % 1) Load Min_EV for all algorithms and concatenate all trials
        % ==============================================================
        MinEV_all = inf(num, nTotalTrials);
        algID     = zeros(1, nTotalTrials);

        colStart = 1;
        for a = 1:nAlg
            X = algs(a).getT(f);
            MinEV = parseBCSOPData(X, trial, num, algs(a).name, f, epsEV);

            cols = colStart:(colStart + trial - 1);
            MinEV_all(:, cols) = MinEV;
            algID(cols) = a;
            colStart = colStart + trial;
        end

        finalEV = MinEV_all(end, :);

        for a = 1:nAlg
            idxA = (algID == a);
            vals = finalEV(idxA);
            valsFinite = vals(isfinite(vals) & ~isnan(vals));
            Data.NumFiniteFinalEV(f,a) = numel(valsFinite);
            if ~isempty(valsFinite)
                Data.MeanFinalEV(f,a) = mean(valsFinite);
                Data.BestFinalEV(f,a) = min(valsFinite);
                Data.WorstFinalEV(f,a) = max(valsFinite);
            end
        end

        % ==============================================================
        % 2) Pairwise accuracy and speed comparisons among all trials
        % ==============================================================
        accTrial   = zeros(1, nTotalTrials);
        speedTrial = zeros(1, nTotalTrials);

        for p = 1:(nTotalTrials - 1)
            for q = (p + 1):nTotalTrials

                % ------------------------------------------------------
                % 2a) Accuracy score A
                % Smaller final Min_EV is better.
                % ------------------------------------------------------
                cmpA = compareSmallerIsBetter(finalEV(p), finalEV(q), epsEV);
                [accTrial(p), accTrial(q)] = addPairPoint(accTrial(p), accTrial(q), cmpA);

                % ------------------------------------------------------
                % 2b) Speed score S
                % Reference level = worse of the two final EVs.
                % The trial that reaches this level earlier is faster.
                % ------------------------------------------------------
                theta = maxFinite(finalEV(p), finalEV(q));

                tauP = firstReach(MinEV_all(:,p), theta, epsEV);
                tauQ = firstReach(MinEV_all(:,q), theta, epsEV);

                if tauP < tauQ
                    speedTrial(p) = speedTrial(p) + 1;
                elseif tauQ < tauP
                    speedTrial(q) = speedTrial(q) + 1;
                else
                    speedTrial(p) = speedTrial(p) + 0.5;
                    speedTrial(q) = speedTrial(q) + 0.5;
                end
            end
        end

        Data.PairwiseAccuracyTrialScore{f} = accTrial;
        Data.PairwiseSpeedTrialScore{f}    = speedTrial;

        % ==============================================================
        % 3) Aggregate trial points to algorithm-level scores
        % ==============================================================
        for a = 1:nAlg
            idxA = (algID == a);
            ac1(f,a) = sum(accTrial(idxA));
            sr1(f,a) = sum(speedTrial(idxA));
        end

        srTot = ac1(f,:) + sr1(f,:);
        score(f,:)    = srTot;
        srTotAll(f,:) = srTot;

        % Larger score is better; rank 1 is best.
        SR(f,:) = tieSafeRanksDesc(srTot, epsEV);
    end

    % ==============================================================
    % 4) Build output tables with sum row
    % ==============================================================
    scoreTbl = array2table([(1:pro)' score], 'VariableNames', tableVarNames);
    rankTbl  = array2table([(1:pro)' SR],    'VariableNames', tableVarNames);

    scoreTbl = addSumRow(scoreTbl);
    rankTbl  = addSumRow(rankTbl);

    Data.Accuracy = ac1;
    Data.Speed = sr1;
    Data.TotalScore = score;
    Data.ScoreTable = scoreTbl;
    Data.RankTable = rankTbl;
end

% =====================================================================
% Helper: parse bound-constrained SOP data from struct or numeric matrix
% =====================================================================
function MinEV = parseBCSOPData(X, trial, num, algName, f, epsEV)

    if isstruct(X)
        evField = findFirstField(X, {'Min_EV','MinEV','EV','Obj','obj','Fitness','fitness','error','Error'});
        if isempty(evField)
            error('Algorithm %s, F%d: struct output must contain Min_EV/MinEV/EV/Obj/Fitness.', algName, f);
        end
        MinEV = X.(evField);

    else
        if ~isnumeric(X)
            error('Algorithm %s, F%d: loader output must be a struct or numeric matrix.', algName, f);
        end

        % Official files may contain an FE column first:
        % [FEs, run1, run2, ..., run25]
        if size(X,2) >= trial + 1 && looksLikeFEColumn(X(:,1))
            MinEV = X(:, 2:(trial+1));
        else
            MinEV = X(:, 1:trial);
        end
    end

    if size(MinEV,1) < num
        error('Algorithm %s, F%d: Min_EV has fewer than num=%d rows.', algName, f, num);
    end
    if size(MinEV,2) < trial
        error('Algorithm %s, F%d: Min_EV has fewer than trial=%d columns.', algName, f, trial);
    end

    MinEV = MinEV(1:num, 1:trial);

    % Precision cutoff and numerical cleaning.
    MinEV(abs(MinEV) < epsEV) = 0;
    MinEV(~isfinite(MinEV)) = Inf;
end

% =====================================================================
% Helper: detect whether the first numeric column is an FE column
% =====================================================================
function tf = looksLikeFEColumn(x)
    x = x(:);
    x = x(isfinite(x));
    if numel(x) < 3
        tf = false;
        return;
    end
    dx = diff(x);
    tf = all(dx >= 0) && x(end) > x(1);
end

% =====================================================================
% Helper: find first available field name
% =====================================================================
function fieldName = findFirstField(S, candidates)
    fieldName = '';
    for i = 1:numel(candidates)
        if isfield(S, candidates{i})
            fieldName = candidates{i};
            return;
        end
    end
end

% =====================================================================
% Helper: compare smaller-is-better values
% cmp = -1 means x is better
% cmp =  1 means y is better
% cmp =  0 means tie
% =====================================================================
function cmp = compareSmallerIsBetter(x, y, tol)
    if nargin < 3 || isempty(tol)
        tol = 1e-12 * max(1, max(abs(x), abs(y)));
    end

    if isnan(x), x = Inf; end
    if isnan(y), y = Inf; end

    localTol = max(tol, 1e-12 * max(1, max(abs(x), abs(y))));

    if isinf(x) && isinf(y)
        cmp = 0;
    elseif abs(x - y) <= localTol
        cmp = 0;
    elseif x < y
        cmp = -1;
    else
        cmp = 1;
    end
end

% =====================================================================
% Helper: add pairwise point based on comparison result
% =====================================================================
function [sp, sq] = addPairPoint(sp, sq, cmp)
    if cmp < 0
        sp = sp + 1;
    elseif cmp > 0
        sq = sq + 1;
    else
        sp = sp + 0.5;
        sq = sq + 0.5;
    end
end

% =====================================================================
% Helper: max for possibly non-finite final values
% =====================================================================
function theta = maxFinite(x, y)
    if isnan(x), x = Inf; end
    if isnan(y), y = Inf; end
    theta = max(x, y);
end

% =====================================================================
% Helper: first sampling point where a curve reaches a reference threshold
% Smaller value is better.
% =====================================================================
function tau = firstReach(curve, theta, tol)
    if isnan(theta) || ~isfinite(theta)
        tau = Inf;
        return;
    end
    idx = find(isfinite(curve) & (curve <= theta + tol), 1, 'first');
    if isempty(idx)
        tau = Inf;
    else
        tau = idx;
    end
end

% =====================================================================
% Helper: tie-safe ranks for descending scores
% Larger score is better; rank 1 is best.
% =====================================================================
function ranks = tieSafeRanksDesc(vals, tol)
    vals = vals(:)';
    n = numel(vals);
    ranks = zeros(1,n);

    [sVals, ord] = sort(vals, 'descend');

    i = 1;
    while i <= n
        j = i;
        while j < n && isEqualTol(sVals(j), sVals(j+1), tol)
            j = j + 1;
        end
        ranks(ord(i:j)) = mean(i:j);
        i = j + 1;
    end
end

% =====================================================================
% Helper: tolerance equality
% =====================================================================
function tf = isEqualTol(x, y, tol)
    if nargin < 3 || isempty(tol)
        tol = 1e-12 * max(1, max(abs(x), abs(y)));
    end
    localTol = max(tol, 1e-12 * max(1, max(abs(x), abs(y))));
    if isinf(x) && isinf(y)
        tf = true;
    else
        tf = abs(x - y) <= localTol;
    end
end

% =====================================================================
% Helper: add sum row to a numeric table with first column Function
% =====================================================================
function T = addSumRow(T)
    last = T(1,:);
    for c = 1:width(T)
        if isnumeric(T{1,c})
            last{1,c} = NaN;
        end
    end
    last.Function = NaN;
    last{1,2:end} = sum(T{:,2:end}, 1, 'omitnan');
    T = [T; last];
end
