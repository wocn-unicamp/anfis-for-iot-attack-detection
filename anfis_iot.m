function [successful, success_rate] = get_success(output, y_test)
    successful = sum(output == y_test);
    success_rate = sprintf("%d %%", round((successful / size(y_test, 1)) * 100));
end

function shared_data = success_res(shared_data)
    y_test = shared_data.df_test(:, end - shared_data.labels + 1:end);
    matches = all(shared_data.output == y_test, 2);
    rMatch = sum(matches);
    total_rows = size(y_test, 1);
    success_rate = sprintf('%d %%', round((rMatch / total_rows) * 100));
    new_row = {'OVERALL', rMatch, success_rate, 2 * 6};
    shared_data.results = [shared_data.results; new_row];
end

function trainfis = train_neuron(fis, epochs, X_train, y_train)
    [in, out, ~] = getTunableSettings(fis);
    opt = tunefisOptions("Method", "anfis", "OptimizationType", "tuning");
    opt.MethodOptions.EpochNumber = epochs;
    opt.Display = "none";
    trainfis = tunefis(fis, [in; out], X_train, y_train, opt);
end

function fis = fuzzycm_fis(clusters, X_train, y_train)
    persistent opts
    if isempty(opts)
        opts = genfisOptions('FCMClustering');
        opts.Verbose = false;
    end
    opts.NumClusters = clusters;
    fis = genfis(X_train, y_train, opts);
end

function [output_col, fis, shared_data] = evaluate_fis(fis, row_name, epochs, n, X_train, y_train, X_test, y_test, shared_data)
    fis = train_neuron(fis, epochs, X_train, y_train);
    sal = evalfis(fis, X_test);
    output_col = double(sal > 0.5);
    shared_data.output(:, n) = output_col;
    [successful, success_rate] = get_success(output_col, y_test);
    rules = showrule(fis);
    total_rules = size(rules, 1);
    new_row = {row_name, successful, success_rate, total_rules};
    shared_data.results = [shared_data.results; new_row];
end

function shared_data = fuzzycm(epoch, clusters, shared_data)
    labels = shared_data.labels;
    features = shared_data.features;
    output = zeros(size(shared_data.df_test, 1), labels);
    cols_to_remove = features - labels + (1:labels);
    X_train_all = shared_data.df_train(:, 1:features);
    X_test_all = shared_data.df_test(:, 1:features);

    for n = 1:labels
        X_train = X_train_all(:, ~ismember(1:features, cols_to_remove(n)));
        y_train = shared_data.df_train(:, cols_to_remove(n));
        X_test = X_test_all(:, ~ismember(1:features, cols_to_remove(n)));
        y_test = shared_data.df_test(:, cols_to_remove(n));
        fis = fuzzycm_fis(clusters, X_train, y_train);
        [output(:, n), fis, shared_data] = evaluate_fis(fis, "FCM", epoch, n, X_train, y_train, X_test, y_test, shared_data);
    end

    shared_data.output = output;
    shared_data.fis = fis;
    shared_data = success_res(shared_data);
end
