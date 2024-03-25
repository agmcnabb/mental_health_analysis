# imports
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import OneHotEncoder


# function to get code when only name is available
def get_variable_code(variable_label):

    # get raw data
    data = pd.read_csv('../alex_tracking_system/dict_csv.csv')

    data_filtered = data[data['Variable label'].str.contains(variable_label, regex=False)]

    try:
        data_filtered_s = data_filtered.iloc[0]
    except:
        return 'VARIABLE NOT FOUND'

    return data_filtered_s['Variable name']


# function to replace negative values with modal
def replace_missing_values(data, variable_code):
    '''
    EXPECTS:
        data: a cm interview dataframe with variable codes for columns and unprocessed values.
        variable_code: a string containing the variable code to replace.

    RETURNS:
        data: a cm interview where the specified column has had missing values replaced.
    '''
    # get modal value
    modal_value = data[variable_code].value_counts().index[0]

    # replace negative values with modal
    data.loc[data[variable_code] < 0, variable_code] = modal_value

    return data


# categorical_list_highest_num_not_true
# def categorical_list_highest_num_not_true(data, cols):
#     '''
#     EXPECTS:
#         data: a cm interview dataframe with variable codes for columns and unprocessed values.
#         cols: a list of variable codes to be processed.
#     RETURNS:
#         output: a cm interview where the specified columns have be transformed.
#     '''
#     output = data.copy()

#     for col in cols:
#         # define scaler
#         scaler = MinMaxScaler(feature_range=(0, 1))

#         # replace missing values
#         output = replace_missing_values(output, col)

#         # turn values negative so that order is reversed
#         output.loc[:, col] = output[col] * -1

#         # apply minmax scaler
#         scaler.fit(output[[col]])
#         output.loc[:, col] = scaler.transform(output[[col]])

#     return output


# DS_numerical
def numerical(data, cols):
    '''
    EXPECTS:
        data: a cm interview dataframe with variable codes for columns and unprocessed values.
        cols: a list of variable codes to be processed.
    RETURNS:
        output: a cm interview where the specified columns have be transformed.
    '''
    output = data.copy()

    for col in cols:
        # define scaler
        scaler = MinMaxScaler(feature_range=(0, 1))

        # replace missing values
        output = replace_missing_values(output, col)

        # apply scaler
        scaler.fit(output[[col]])
        output.loc[:, col] = scaler.transform(output[[col]])

    return output

# DS_cat_nominal
def cat_nominal(data, cols):
    '''
    EXPECTS:
        data: a cm interview dataframe with variable codes for columns and unprocessed values.
        cols: a list of variable codes to be processed.
    RETURNS:
        output: a cm interview where the specified columns have be transformed.
    '''
    output = data.copy()

    # define encoder
    encoder = OneHotEncoder(handle_unknown='ignore', sparse=False, drop='first')

    # replace missing values
    for col in cols:
        output = replace_missing_values(output, col)

    # apply encoder
    encoder.fit(output[cols])
    encoded_data = encoder.transform(output[cols])

    encoded_df = pd.DataFrame(encoded_data, columns=encoder.get_feature_names_out(cols))

    output_final = pd.concat([output, encoded_df], axis=1).drop(columns=cols)

    return output_final


# DS_cat_nominal_binary_Y1_N0
def cat_nominal_binary_Y1_N0(data, cols):
    '''
    EXPECTS:
        data: a cm interview dataframe with variable codes for columns and unprocessed values.
        cols: a list of variable codes to be processed.
    RETURNS:
        output: a cm interview where the specified columns have be transformed.
    '''
    output = data.copy()

    for col in cols:
        # replace missing values
        output = replace_missing_values(output, col)

    return output


# DS_cat_nominal_binary_Y1_N2
def cat_nominal_binary_Y1_N2(data, cols):
    '''
    EXPECTS:
        data: a cm interview dataframe with variable codes for columns and unprocessed values.
        cols: a list of variable codes to be processed.
    RETURNS:
        output: a cm interview where the specified columns have be transformed.
    '''
    output = data.copy()

    for col in cols:
        # replace missing values
        output = replace_missing_values(output, col)

        # replace any remaing values that aren't 1
        output.loc[output[col] != 1, col] = 0

    return output


# DS_cat_ordinal_highest_num_is_highest_value
def cat_ordinal_highest_num_is_highest_value(data, cols):
    '''
    EXPECTS:
        data: a cm interview dataframe with variable codes for columns and unprocessed values.
        cols: a list of variable codes to be processed.
    RETURNS:
        output: a cm interview where the specified columns have be transformed.
    '''
    output = data.copy()

    for col in cols:
        # define scaler
        scaler = MinMaxScaler(feature_range=(0, 1))

        # replace missing values
        output = replace_missing_values(output, col)

        # apply minmax scaler
        scaler.fit(output[[col]])
        output.loc[:, col] = scaler.transform(output[[col]])


    return output


# DS_cat_ordinal_lowest_num_is_highest_value
def cat_ordinal_lowest_num_is_highest_value(data, cols):
    '''
    EXPECTS:
        data: a cm interview dataframe with variable codes for columns and unprocessed values.
        cols: a list of variable codes to be processed.
    RETURNS:
        output: a cm interview where the specified columns have be transformed.
    '''
    output = data.copy()

    for col in cols:
        # define scaler
        scaler = MinMaxScaler(feature_range=(0, 1))

        # replace missing values
        output = replace_missing_values(output, col)

        # turn values negative so that order is reversed
        output.loc[:, col] = output[col] * -1

        # apply minmax scaler
        scaler.fit(output[[col]])
        output.loc[:, col] = scaler.transform(output[[col]])

    return output


function_dict = {
    "DS_numerical": numerical,
    "DS_cat_nominal": cat_nominal,
    "DS_cat_ordinal_lowest_num_is_highest_value": cat_ordinal_lowest_num_is_highest_value,
    "DS_cat_ordinal_highest_num_is_highest_value": cat_ordinal_highest_num_is_highest_value,
    "DS_cat_nominal_binary_Y1_N2": cat_nominal_binary_Y1_N2,
    "DS_cat_nominal_binary_Y1_N0": cat_nominal_binary_Y1_N0
}


# use the above functions to process given features
def feature_processor(data, var_cats):
    '''
    EXPECTS:
        data: a cm interview dataframe with variable codes for columns and unprocessed values.
        var_cats: a dict whose keys are the feature categories and values are
            lists of feature names within each category.
    RETURNS:
        output: a cm interview where the features have be transformed.
    '''
    output = data.copy()

    # for each feature category
    for cat in var_cats:
        # select all features in that category
        features = var_cats[cat]

        # convert variable labels into codes
        var_codes = [get_variable_code(feature) for feature in features]

        # and pass those codes into the appropriate function
        output = function_dict[cat](output, var_codes)

    return output


def add_smfq_label(raw_data):
    '''This function processes a sweep 6 cm interview dataframe.
    It adds a column 'smfq_label' that indicates whether the YP indicates signs
    of depression based on the answers to the Short Moods and Feelings
    Questionaire (SMFQ).
    It also removes these features once the label has been add.

    EXPECTS:
        A dataframe with unprocessed values (e.g. -9, -8, -1, 1, 2, 3)
        and raw variable names (e.g. 'FCMDSA00')

    RETURNS:
        A dataframe with the SMFQ features removed and an smfq_label column added.
    '''
    # define smfq variables
    smfq_variables = [
        'FCMDSA00',
        'FCMDSB00',
        'FCMDSC00',
        'FCMDSD00',
        'FCMDSE00',
        'FCMDSF00',
        'FCMDSG00',
        'FCMDSH00',
        'FCMDSI00',
        'FCMDSJ00',
        'FCMDSK00',
        'FCMDSL00',
        'FCMDSM00'
    ]

    # filter data for the smfq questions
    smfq_questions = raw_data.loc[:, smfq_variables]

    # replace missing data with the modal value
    data_to_impute = smfq_questions.copy()

    for var in smfq_variables:
        data_to_impute = replace_missing_values(data_to_impute, var)

    smfq_questions_imputed = data_to_impute.copy()

    # create a new column with the YP's total SMFQ score
    smfq_map = {
        1: 0,
        2: 1,
        3: 2
    }

    final_scores = []
    for i, yp in smfq_questions_imputed.iterrows():
        final_score = 0
        for feature, value in yp.items():
            final_score += smfq_map[value]

        final_scores.append(final_score)

    smfq_questions_imputed_with_finals = smfq_questions_imputed.copy()

    smfq_questions_imputed_with_finals['smfq_final_score'] = final_scores

    # create a new column with a depressed/not depressed label based on a threshold of 12
    smfq_labels = []
    for i, yp in smfq_questions_imputed_with_finals.iterrows():
        smfq_labels.append(0 if yp.smfq_final_score < 12 else 1)

    smfq_questions_imputed_with_finals_and_threshold = smfq_questions_imputed_with_finals.copy()

    smfq_questions_imputed_with_finals_and_threshold['smfq_label'] = smfq_labels

    # create an output df that drops the smfq questions and adds the final label
    X = raw_data.copy()

    # drop raw smfq columns
    X.drop(columns=smfq_variables, inplace=True)

    # define label
    y = smfq_questions_imputed_with_finals_and_threshold['smfq_label']

    return X, y
