%%%%%%%%%%%%%%%%%%%%%%%%%%% Top level test functions. %%%%%%%%%%%%%%%%%%%%

function testEstimators(R)
    % Default parameters.
    if ~exist('R', 'var'), R = 1e6; end
    
    % Clear the output Latex file.
    fid=fopen('tables.tex','w');
    fclose(fid);
    
    % Test on the normal distributions.
    expNames = {}; overallREs = {};
    for d = [3, 4]
        for rho = [-0.25, 0, 0.5, 0.75]
            allREs = TestEstimatorsNormal(d, rho, R);
            expNames{end+1} = sprintf('d=%d, rho=%.2g', d, rho);
            overallREs{end+1} = allREs;
            fprintf('\n\n');
        end
    end
    
    PrintIndividualComparisonTables(expNames, overallREs);
    
%     % Test the Laplace distribution.
%     TestEstimatorsLaplace(R);
end

function [allREs] = TestEstimatorsLaplace(R)
    fprintf('Testing on Laplace distribution (d=4)\n\n');

    % These can't be changed w/o updating ExtractNormalTrueValues.
    d = 4; gammas = [6, 8, 10, 12];

    % The actual probabilities we are trying to estimate.
    trueVals = [0.00040930368933106436, 0.000024348718662409335, 1.4417010595294805e-6, 8.525527399477557e-8];

    experimentName = sprintf(...
        'Multivariate Laplace with $d=%d$', d);
    
    sampleUncond = @(R) ...
    	(sqrt(exprnd(1,[R,1]))*ones([1,d])).*normrnd(0,1,[R,d]);

    sampleCondUni = @(gamma, R) FirstISLaplace(gamma, d, R);
    
    OneProb = @(x) (1/2)*exp(-sqrt(2)*x); 
    TwoProbConst = [6.166523748515595e-7, 1.0069543881730026e-8, 1.6808472359099376e-10, 2.8475951963393738e-12];
    TwoProb = @(x) TwoProbConst(find(gammas == x, 1));
        
    allREs = TestIndividualDistribution(gammas, d, R, trueVals, ...
        experimentName, sampleUncond, sampleCondUni, [], OneProb, TwoProb);
end

function [allREs] = TestEstimatorsNormal(d, rho, R)
    fprintf('Testing on normal distribution (d=%d and rho=%.2g)\n\n',...
                d, rho);

    % These can't be changed w/o updating the 'true values'.
    sigma = 1; gammas = [2, 4, 6, 8];

    % Construct covariance matrix (n.b. we assume mu = zeros([1,d])).
    sigma = 1; Sigma = sigma^2 .* ((1 - rho)*eye(d) + rho);

    experimentName = sprintf(...
            'Multivariate normal with $d=%d$ and $\\rho=%.2g$', d, rho);
    
    sampleUncond = @(R) ...
        mvnrnd(zeros([1,d]), Sigma, R);

    sampleCondUni = @(gamma, R) ...
        FirstISNormal(gamma, Sigma, R);

    sampleCondBi = @(gamma, R) ...
        SecondISNormal(gamma, Sigma, R);

    % Construct the numerical integration functions.
    integrand = @(x,y) mvnpdf([x,y],[0,0],[[1,rho];[rho,1]]);
    mvn_integrand = @(x1, x2) reshape(integrand(x1(:), x2(:)), size(x1));
    
    OneProb = @(x) normcdf(x, 0, sigma, 'upper');
    TwoProb = @(x) integral2(mvn_integrand, x, Inf, x, Inf);

    % The actual probabilities we are trying to estimate.
    trueVals = ExtractNormalTrueValues(d, rho);
   
    allREs = TestIndividualDistribution(gammas, d, R, trueVals, ...
        experimentName, sampleUncond, sampleCondUni, sampleCondBi, OneProb, TwoProb);
end

function [allREs] = TestIndividualDistribution(gammas, d, R, trueVals, ...
        experimentName, sampleUncond, sampleCondUni, sampleCondBi, ...
        OneProb, TwoProb)

    names = {'$\hat{\alpha}_0$', ...
        '$\overline{\alpha}$', ...
        '$\overline{\alpha}{-}q$', ...
        '$\hat{\alpha}_1$', ...
        '$\hat{\alpha}_2$', ...
        '$\hat{\alpha}_1^{[1]}$', ...
        '$\hat{\alpha}_2^{[2]}$', ... 
        '$\widehat{(\beta_1 \ddagger \alpha)}$', ...
        '$\widehat{(\beta_2 \ddagger \alpha)}$'}; 
    
    estimators = {...
        @() CMC(gammas, R, sampleUncond), ...
        @() FirstOrderIEF(gammas, d, OneProb), ...
        @() SecondOrderIEF(gammas, d, OneProb, TwoProb), ...
        @() Alpha1Raw(gammas, d, R, sampleUncond, OneProb), ...
        @() Alpha2Raw(gammas, d, R, sampleUncond, OneProb, TwoProb), ...
        @() Alpha1FirstIS(gammas, d, R, sampleCondUni, OneProb), ...
        @() Alpha2SecondIS(gammas, d, R, sampleCondBi, OneProb, TwoProb), ...
        @() BetaFirstIS(gammas, d, R, sampleCondUni, OneProb), ...
        @() BetaSecondIS(gammas, d, R, sampleCondBi, OneProb, TwoProb)};
        
    if isempty(sampleCondBi)
        names([7, 9]) = [];
        estimators([7,9]) = [];
    end
    
    allEsts = NaN([length(estimators), length(gammas)]);
    allREs = NaN([length(estimators), length(gammas)]);
    allEstREs = NaN([length(estimators), length(gammas)]);
    allStds = NaN([length(estimators), length(gammas)]);
    allTimes = NaN([length(estimators), 1]);
    
    % Loop through all the estimators, run them and store the results.
    for estNum = 1:length(estimators)
        rng(0);
        fprintf('%s...\n', names{estNum}); tic;
        [ests, stds] = estimators{estNum}();
        allTimes(estNum) = toc;
        allEsts(estNum, :) =  ests;
        allStds(estNum, :) = stds;
        allREs(estNum, :) = abs(ests - trueVals) ./ trueVals;
        if ~contains(names{estNum}, 'Beta')
            allEstREs(estNum, :) = stds ./ (ests * sqrt(R));
        end
    end
    
    tableNames = {'Estimates', 'Relative Errors', ...
        'Estimated Relative Errors', 'Standard Deviations'};
    tableDatas = {allEsts, allREs, allEstREs, allStds};
    
    PrintResults(experimentName, names, gammas, tableNames, ...
        tableDatas, allTimes, trueVals)
end

%%%%%%%%%%%%%%%% Support functions for the testing system. %%%%%%%%%%%%%%%%

function [vals] = ExtractNormalTrueValues(d, rho) 
    trueValues = {};
    validRho = [-0.25, 0, 0.5, 0.75];
    if ~(d == 3 || d == 4)
        error('Please select d as either 3 or 4');
    end
    if isempty(find(validRho==rho, 1))
        error('Please select rho as one of: %s', ...
            num2str(validRho, '%1.2g'));
    end

    trueValues{3, validRho==-0.25} = [0.06799091186516087,0.00009501371677945498,2.959762935113072e-9,1.8662881722815216e-15];
    trueValues{3, validRho==0} = [0.06670946508530937,0.00009501071632845022,2.959762932193018e-9,1.8662881722815204e-15];
    trueValues{3, validRho==0.5} = [0.05746654850547127,0.00009359498874273782,2.958599678113989e-9,1.866282808003815e-15];
    trueValues{3, validRho==0.75} = [0.047687590961441886,0.00008557247917455595,2.9031222164904503e-9,1.8623228371618052e-15];
    trueValues{4, validRho==-0.25} = [0.09048157057617723,0.00012668494989267025,3.946350580150755e-9,2.4883842297086954e-15];
    trueValues{4, validRho==0} = [0.08794194790060532,0.00012667894905419653,3.9463505743106474e-9,2.488384229708693e-15];
    trueValues{4, validRho==0.5} = [0.07154940320706243,0.00012392437484362077,3.944033444787854e-9,2.4883735045427403e-15];
    trueValues{4, validRho==0.75} = [0.056331851376941765,0.00010953627449359208,3.838057315756161e-9,2.4805896249322163e-15];

    vals = trueValues{d, validRho==rho};
end

function PrintResults(expName, names, gammas, tableNames, ...
        tableDatas, allTimes, trueVals)
    
    gammasStr = arrayfun(...
        @(x) sprintf('$\\gamma = %d$', x), gammas, 'UniformOutput', false);

    input.dataFormat = {'%1.3e'};
    input.tablePlacement = 'H';
    input.dataNanString = '*';
    input.tableColLabels = gammasStr;
    fid=fopen('tables.tex','a');

    % For each output table.
    for t = 1:length(tableNames)
        % Find which kind of table we are printing.
        tableName = tableNames{t};
        
        % Skip the estimated RE table (nothing interesting there).
        if strcmp(tableName, 'Estimated Relative Errors')
            continue;
        end
        
        % Print a header.
        tableData = tableDatas{t};
        fprintf('\n\n%s\n\n', tableName);
        
        % Show 'degenerated' values as NaN.
        tableData(tableDatas{end} == 0) = NaN;
 
        % Print the row of gammas.
        fprintf('%20s:\t', 'gamma'); fprintf('%1.2f\t\t', gammas);
        fprintf('\n');
             
        % Prepend the 'true values' to the estimates table.
        if strcmp(tableName, 'Estimates')
            % Print them.
            fprintf('%20s:\t', '$\alpha$');
            fprintf('%1.3e\t', trueVals); 
            fprintf('\n');
            
            % Add the the table row labels.
        	input.tableRowLabels = {'$\alpha$', names{:}};
            
            % Add them to the table.
            tableData = [trueVals; tableData];
        else
            input.tableRowLabels = names;
        end

        % Print the output for each estimator.
        for i = 1:length(names)
            if all(isnan(tableData(i,:)))
                continue
            end
            fprintf('%20s:\t', names{i});
            fprintf('%1.3e\t', tableData(i,:)); 
            fprintf('\n');
        end
    
        % Create a LaTeX table and write it to the tables.tex file.
        input.data = tableData;
        input.tableCaption = sprintf('%s %s', expName, tableName);
        latex = latexTable(input);
        [nrows,~] = size(latex);
        for row = 1:nrows
            fprintf(fid,'%s\n',latex{row,:});
        end
        fprintf(fid,'\n\n');
    end
    
    fprintf('\n\nTimes! \n\n');    
    for i = 1:length(names)
        fprintf('%20s:\t', names{i});
        fprintf('%1.3f\t', allTimes(i)); 
        fprintf('\n');
    end
    
    fclose(fid);
end

function PrintIndividualComparisonTables(expNames, overallREs)
    % (This is not so nice: I shouldn't define these variables twice...)
    gammas = [2, 4, 6, 8];
    names = {'$\hat{\alpha}_0$', ...
        '$\overline{\alpha}$', ...
        '$\overline{\alpha}{-}q$', ...
        '$\hat{\alpha}_1$', ...
        '$\hat{\alpha}_2$', ...
        '$\hat{\alpha}_1^{[1]}$', ...
        '$\hat{\alpha}_2^{[2]}$', ...
        '$\widehat{(\beta_1 \ddagger \alpha)}$', ...
        '$\widehat{(\beta_2 \ddagger \alpha)}$'}; 

    % Get the strings for the column headers.
    gammasStr = arrayfun(...
        @(x) sprintf('$\\gamma = %d$', x), gammas, 'UniformOutput', false);


    inds = [4, 2; 5, 3; 6, 4; 7, 5; 8, 6; 9, 7];
    
    expNames{end+1} = 'Average';
    
    input.tableColLabels = gammasStr;
    input.tableRowLabels = expNames;
    input.dataFormat = {'%1.3g'};
    input.tablePlacement = 'H';
    input.dataNanString = '*';
    fid=fopen('tables.tex','a');

    for i = 1:size(inds, 1)
        ratios = cellfun(@(X) X(inds(i,1),:) ./ X(inds(i,2),:),...
            overallREs, 'UniformOutput', false);
        ratios = vertcat(ratios{:}); 
        input.data = vertcat(ratios, mean(ratios));

        input.tableCaption = sprintf(...
            'Ratio of %s''s relative error to %s''s', ...
            names{inds(i,1)}, names{inds(i,2)});
        latex = latexTable(input);
        [nrows,~] = size(latex);
        for row = 1:nrows
            fprintf(fid,'%s\n',latex{row,:});
        end
        fprintf(fid,'\n\n');
    end

    fclose(fid);
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Estimators %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Crude Monte Carlo
function [ests, stds] = CMC(gammas, R, sampleUncond)
    Xs = sampleUncond(R);
    ests = zeros(size(gammas));
    stds = zeros(size(gammas));
    M = max(Xs, [], 2);
    for g = 1:length(gammas)
        ests(g) = mean(M > gammas(g));
        stds(g) = std(M > gammas(g));
    end
end

% First order IEF approximation.
function [ests, stds] = FirstOrderIEF(gammas, d, OneProb)
    ests = d*OneProb(gammas);
    stds = NaN(size(ests));
end

% Second order IEF approximation.
function [ests, stds] = SecondOrderIEF(gammas, d, OneProb, TwoProb)
    second = zeros(size(gammas));
    for g = 1:length(gammas)
        second(g) = TwoProb(gammas(g));
    end
    ests = d*OneProb(gammas) - nchoosek(d, 2) * second;
    stds = NaN(size(ests));
end

% Our raw alpha_1 estimator.
function [ests, stds] = Alpha1Raw(gammas, d, R, sampleUncond, OneProb)
    Xs = sampleUncond(R);
    ests = zeros(size(gammas));
    stds = zeros(size(gammas));
    for g = 1:length(gammas)
        E = sum(Xs > gammas(g), 2);
        ests(g) = d*OneProb(gammas(g)) - mean((E-1).*(E>=2));
        stds(g) = std((E-1).*(E>=2));
    end
end

% Our raw alpha_2 estimator.
function [ests, stds] = Alpha2Raw(gammas, d, R, sampleUncond, OneProb, TwoProb)
    Xs = sampleUncond(R);   
    ests = zeros(size(gammas));
    stds = zeros(size(gammas));
    for g = 1:length(gammas)
        E = sum(Xs > gammas(g), 2);
        ests(g) = d*OneProb(gammas(g)) -...
            nchoosek(d,2)*TwoProb(gammas(g)) + ...
            mean((1 - E + 0.5*E.*(E-1)).*(E>=3));
        stds(g) = std((1 - E + 0.5*E.*(E-1)).*(E>=3));
    end
end

% Our alpha_1 estimator with first-order IS.
function [ests, stds] = Alpha1FirstIS(gammas, d, R, sampleFirstIS, OneProb)
    ests = zeros(size(gammas));
    stds = zeros(size(gammas));
    
    for g = 1:length(gammas)
        % Simulate vectors from the IS distribution.
        [Xi, rest] = sampleFirstIS(gammas(g), R);
        
        % Stick the result together and estimate.
        Xs = [Xi, rest];
        E = sum(Xs > gammas(g), 2);
        
        ests(g) = d*OneProb(gammas(g)) * mean(1 ./ E);
        stds(g) = d*OneProb(gammas(g)) * std(1 ./ E);
    end
end

% Our 'simple' beta estimator (applied to the 'alpha' problem) which
% requires the first-order IS sampler. This version does not assume the
% distribution is exchangable, and tries to match the number random
% variables sampled to be R*d.
%         @() BetaFirstIS(gammas, Sigma, R, OneProb, true), ...
%         @() BetaFirstIS(gammas, Sigma, R, OneProb, false), ...
function [ests, stds] = BetaFirstIS(gammas, d, R, sampleFirstIS, OneProb, ...
        constY)
    if ~exist('constY', 'var'), constY = true; end
    
    % The total number of X vectors to simulate (noting that we don't use
    % all the components from all the vectors, so this is larger than R).
    if constY
        RAdj = ceil(2*d*R/(d+2));
        split = ceil(RAdj / (d-1)); 
    else
        RAdj = R; split = ceil(RAdj / d);
    end
    
    ests = zeros(size(gammas));
    vars = zeros(size(gammas));
    
    for g = 1:length(gammas)
        [Xi, rest] = sampleFirstIS(gammas(g), RAdj);

        ests(g) = OneProb(gammas(g));
        for i = 2:d
            if constY
                r1 = (i-2)*split + 1; r2 = min((i-1)*split, RAdj);
            else
                r1 = (i-1)*split + 1; r2 = min(i*split, RAdj);
            end
            XsExtract = zeros(length(r1:r2), d);
            XsExtract(:, i) = Xi(r1:r2);
            XsExtract(:, setdiff(1:d,i)) = rest(r1:r2,:);
            
            est = OneProb(gammas(g)) * ...
                (sum(XsExtract(:,1:(i-1)) >= gammas(g), 2) == 0);
            ests(g) = ests(g) + mean(est);
            vars(g) = vars(g) + var(est);
        end
    end
    
    stds = sqrt(vars);
end

% Our alpha_2 estimator with second-order IS.
function [ests, stds] = Alpha2SecondIS(gammas, d, R, sampleSecondIS, ...
    OneProb, TwoProb)

    ests = zeros(size(gammas)); stds = zeros(size(gammas));
    
    for g = 1:length(gammas)
        if R >= 1e5
            fprintf('(%d/%d) gamma = %d.\n', g, length(gammas), gammas(g));
        end
        
        % Generate the first two columns conditioned on being > gamma.
        [Xij, rest] = sampleSecondIS(gammas(g), R);
        Xs = [Xij, rest];
        
        E = sum(Xs > gammas(g), 2);
        
        q = nchoosek(d,2) * TwoProb(gammas(g));
        ests(g) = d*OneProb(gammas(g)) - 2*q*mean(1 ./ E);
        stds(g) = 2*q*std(1 ./ E);
    end
end

% Our beta_2 estimator (which uses the second-order IS sampler).
function [ests, stds] = BetaSecondIS(gammas, d, R, sampleSecondIS, ...
        OneProb, TwoProb)

    split = ceil(R / nchoosek(d, 2));
    ests = zeros(size(gammas)); vars = zeros(size(gammas));
    
    for g = 1:length(gammas)
        if R >= 1e5
            fprintf('(%d/%d) gamma = %d.\n', g, length(gammas), gammas(g));
        end
        
        % Generate the first two columns conditioned on being > gamma.
        [Xij, rest] = sampleSecondIS(gammas(g), R);
        
        ests(g) = d*OneProb(gammas(g)); two = TwoProb(gammas(g));
        
        estNum = 0; 
        for i = 1:d
            for j = i+1:d
                estNum = estNum + 1;
                r1 = (estNum-1)*split + 1; r2 = min(estNum*split, R);
                XsExtract = zeros(length(r1:r2), d);
                XsExtract(:, [i,j]) = Xij(r1:r2,:);
                XsExtract(:, setdiff(1:d,[i,j])) = rest(r1:r2,:);
                
                E = sum(XsExtract > gammas(g), 2);
                
                inds = setdiff(1:(j-1), i);
                est = two * (1-E) .* ...
                    (sum(XsExtract(:,inds) >= gammas(g), 2) == 0);
                
                ests(g) = ests(g) + mean(est);
                vars(g) = vars(g) + var(est);
            end
        end
    end
    stds = sqrt(vars);
end

%TODO(Pat): Not a priority.
% Our raw alpha_1 estimator using the optimal control variate constant.
% function [ests, stds] = Alpha1RawCV(gammas, Xs)
%     d = size(Xs, 2);
%     ests = zeros(size(gammas));
%     sds = zeros(size(gammas));
%     for g = 1:length(gammas)
%         E = sum(Xs > gammas(g), 2);
%         ests(g) = 0;
%         sds(g) = 0;
%     end
% %     
% %     W = sum([V,X,Y,Z]>a,2);  %Use the sum (i.e. K) as control variate *)
% %     Zprime = max(max(max(V,X),Y),Z)>a;
% % 
% %     z_hat = mean(Zprime); z_var = var(Zprime);
% %     w_hat = mean(W); w_var = var(W);
% % 
% %     %covariance matrix:
% %     var_ZW = cov(Zprime, W);
% %     alpha = - var_ZW(1,2)/w_var;
% %     alphas(i) = alpha;
% %     
% %     estimated_values(i) = z_hat + alpha*(w_hat - d*OneProb(a));
% %     %The usual measure of efficiency of control variates can be found from
% %     %rho = corrcoef(Z, W); and sqrt((1 - rho(1,2)^2))
% %     rho = corrcoef(Zprime, W); 
% %     sds(i) = std(Zprime)*sqrt((1 - rho(1,2)^2));    
% end

%%%%%%%%%%%%%%%%%%%%% IS Distribution Samplers %%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [Xi, rest] = FirstISNormal(gamma, Sigma, R)
    Sigma11 = Sigma(1:end-1, 1:end-1); Sigma12 = Sigma(1:end-1, end);
    Sigma22 = Sigma(end, end);
    
    % The covariance matrix of X_{-1} conditioned on X_1. 
    CondSigma = Sigma11 - Sigma12 * inv(Sigma22) * (Sigma12');

    % Generate the first column conditioned on being > gamma.
    Xi = ConditionalNormalMinimumUnivariate(gamma, R);

    % Just check nothing funny happened with this numerics.
    if sum(isnan(Xi) | Xi == Inf) > 0
        error('Conditional sampling failed.');
    elseif sum(Xi <= gamma) > 0
        warning('Conditional normals too small.');
        Xi(Xi <= gamma) = gamma + 1e-10; % (to be very safe)
    end
        
    % Calculate the mean of the remaining elements given this X_1.
    CondMu = (Sigma12 * inv(Sigma22) * (Xi'))';
        
    % Simulate the remaining components.
    rest = mvnrnd(CondMu, CondSigma);
end
    
function [Xij, rest] = SecondISNormal(gamma, Sigma, R)
    d = size(Sigma, 1); rho = Sigma(1,2) / Sigma(1,1);
    Sigma11 = Sigma(1:end-2, 1:end-2);
    Sigma12 = Sigma(1:end-2, end-1:end);
    Sigma22 = Sigma(end-1:end, end-1:end);
     
    % The covariance matrix of X_{-1} conditioned on X_1. 
    CondSigma = Sigma11 - Sigma12 * inv(Sigma22) * (Sigma12');
    
    % Generate the first two columns conditioned on being > gamma.
    Xij = ConditionalNormalMinimumBivariate(gamma, rho, R)';

    if sum(sum(isnan(Xij) | Xij == Inf)) > 0
        error('(1) Conditional sampling failed.');
    elseif sum(sum(Xij <= gamma)) > 0
        warning('Conditional normals too small.');
        Xij(Xij <= gamma) = gamma + 1e-10; % (to be very safe)
    end

    % Calculate the mean of the remaining elements given this X_1.
    CondMu = (Sigma12 * inv(Sigma22) * (Xij'))';

    % Simulate the remaining components.
    rest = mvnrnd(CondMu, CondSigma);
end   

function [Xi, rest] = FirstISLaplace(gamma, d, R)
    % Generate X_i | X_i > gamma.
    Xi = exprnd(1/sqrt(2), [R,1]) + gamma;
    
    % Generate the rest.
    rest = randn([R, d-1]);
    for r = 1:R
        if mod(r,round(R/10)) == 0
            fprintf('%d%% Done!\n',100*round(r/R,1))
        end
        pd = makedist('InverseGaussian', 'mu', sqrt(2)*abs(Xi(r)), ...
            'lambda', 2*Xi(r)^2);
        Denom = sqrt(random(pd,1,1)); %generate the denominator as the square root of the GIG distribution
        rest(r, :) = (Xi(r)/Denom) .* rest(r, :);
    end
end

%%%%%%%%%%%%%%%%%%%%% Conditional Samplers %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Simulate X | X > gamma where X has a standard normal distribution.
function [Xs, efficiency] = ConditionalNormalMinimumUnivariate(gamma, R,...
        verbose)
    % Use the AR method in Proposition 2.3 of Robert (1995).
    if ~exist('verbose', 'var'), verbose = false; end
    alphaStar = (gamma+sqrt(gamma^2 + 4))/2;
    rho = @(z) exp(-(z-alphaStar).^2 ./ 2);

    % Use the vectorised way when generating a batch of random variables.
    if R > 10
        expEffic = alphaStar * exp(alphaStar * gamma - (alphaStar^2)/2) *...
            normcdf(-gamma)*sqrt(2*pi);
        if verbose, fprintf('Expected efficiency: %f.\n', expEffic); end

        maxNeededRVs = max(ceil(2*R*(1/expEffic)), 10);
        proposals = gamma + exprnd(1/alphaStar, [maxNeededRVs, 1]);

        uniforms = rand([maxNeededRVs, 1]);
        accept = uniforms <= rho(proposals);
        efficiency = mean(accept);

        Xs = proposals(find(accept==1, R));
        if numel(Xs) < R || sum(Xs == Inf) > 0 || sum(isnan(Xs)) > 0
            error('AR broken!');
        end
    
    % For small R, just use for loops.
    else
        Xs = NaN(R,1); numAttempts = 0; giveUp = 10;
        for i = 1:R
            success = 0;
            for attempt = 1:giveUp
                z = gamma + exprnd(1/alphaStar);
                numAttempts = numAttempts + 1;

                if rand() <= rho(z)
                    success = 1;
                    Xs(i) = z;
                    break
                end           
            end

            if success == 0, error('AR is taking too long'); end

        end
        efficiency = R / numAttempts;
        if verbose, fprintf('Actual efficiency: %f.\n', efficiency); end
    end
end

% Simulate (X,Y) | min(X,Y) > gamma where (X,Y) has a multivariate
% normal distribution and var(X) = var(Y) = 1, cov(X,Y) = cor(X,Y) = rho.
function out = ConditionalNormalMinimumBivariate(gamma, rho, R)
    out = mvrandn(gamma*[1, 1], Inf*[1, 1], [1, rho; rho, 1], R);
end

%%%%%%%%%%%%%%%%%%% Code to print LaTeX tables %%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Source:
%    https://au.mathworks.com/matlabcentral/fileexchange/44274-latextable

