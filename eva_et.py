import pandas as pd

from ensemble_functions import Annotate_word, Run_external_taggers
from process_features import Calculate_normalized_length, Add_code_context


def eva_et(data_path, out_file):
    df = pd.read_csv(data_path)
    identifier_context = 'FUNCTION'
    types = df['TYPE'].values.tolist()
    names = df['DECLARATION'].values.tolist()
    outputs = []

    for i, (identifier_type, identifier_name) in enumerate(zip(types, names)):
        identifier_name = identifier_name.strip()
        identifier_type = identifier_type.strip()
        if 'throws' in identifier_name:
            identifier_name = identifier_name[:identifier_name.find('throws')].strip()
        output = []
        try:
            ensemble_input = Run_external_taggers(identifier_type + ' ' + identifier_name, identifier_context)
            ensemble_input = Calculate_normalized_length(ensemble_input)
            ensemble_input = Add_code_context(ensemble_input,identifier_context)
            for key, value in ensemble_input.items():
                result = Annotate_word(value[0], value[1], value[2], value[3], value[4].value)
                output.append(result)
        except Exception as e:
            print(identifier_name, 'failure')
            output.append('UNK')

        output_str = ' '.join(output)+'\n'
        # print(i, output_str)
        outputs.append(output_str)

    with open(out_file, 'w') as f:
        f.writelines(outputs)


if __name__=='__main__':
    # change the first parameter to "PATH TO ensemble_format DATA",
    # the second parameter "PATH TO OUTPUT FILE"
    eva_et('./dataset/ensemble_format/test_input.csv', 'et_out.txt')