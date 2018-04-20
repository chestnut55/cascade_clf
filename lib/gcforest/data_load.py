import pandas as pd
from sklearn.preprocessing import LabelEncoder
import os


def obesity_data():
    abundance = '../lib/gcforest/data/obesity/abundance_obesity.txt'
    # abundance = 'data/obesity/abundance_obesity.txt'
    f = pd.read_csv(abundance, sep='\t', header=None, index_col=0)
    f = f.T
    f = f.loc[(f != 0).any(axis=1)]

    f.set_index('sampleID', inplace=True)

    l = f['disease'].values
    label_encoder = LabelEncoder()
    integer_encoded = label_encoder.fit_transform(l)

    feature_identifier = 'k__'
    feat = [s for s in f.columns if sum([s2 in s for s2 in feature_identifier.split(':')]) > 0]
    f = f.loc[:, feat].astype('float')
    f = (f - f.min()) / (f.max() - f.min())
    # X_train, X_test, y_train, y_test = train_test_split(f, integer_encoded, test_size=0.1, random_state=42)
    #
    # x_train = X_train.values.reshape(-1, 1, len(X_train.columns)).astype('float32')
    # x_test = X_test.values.reshape(-1, 1, len(X_train.columns)).astype('float32')
    return f, integer_encoded


def cirrhosis_strain_data():
    print(os.path.dirname(os.path.realpath('__file__')))
    abundance = '../lib/gcforest/data/cirrhosis/abundance_cirrhosis_strain.txt'
    # abundance = 'data/cirrhosis/abundance_cirrhosis_strain.txt'
    f = pd.read_csv(abundance, sep='\t', header=None, index_col=0)
    f = f.T
    f = f.loc[(f != 0).any(axis=1)]
    f.set_index('sampleID', inplace=True)

    l = f['disease'].values
    label_encoder = LabelEncoder()
    integer_encoded = label_encoder.fit_transform(l)

    feature_identifier = 'k__'
    feat = [s for s in f.columns if sum([s2 in s for s2 in feature_identifier.split(':')]) > 0]
    print("feaures length:", len(feat), feat)
    f = f.loc[:, feat].astype('float')
    f = (f - f.min()) / (f.max() - f.min())
    return f, integer_encoded


def cirrhosis_genus_data():
    f = pd.read_csv('../lib/gcforest/data/cirrhosis/count_matrix.csv', sep=',', header=None)
    # f = pd.read_csv('data/cirrhosis/abundance_cirrhosis_genus.txt', sep='\t', header=None, index_col=0)
    # f = f.T
    f = f.loc[(f != 0).any(axis=1)]

    l = pd.read_csv('../lib/gcforest/data/cirrhosis/labels_genus.txt', sep='\t', header=None)
    # l = pd.read_csv('data/cirrhosis/labels_genus.txt', sep='\t', header=None)
    l = l.T.iloc[0]
    label_encoder = LabelEncoder()
    integer_encoded = label_encoder.fit_transform(l)
    f = (f - f.min()) / (f.max() - f.min())

    return f, integer_encoded


def hmp_hmpii_data():
    abundance = '../lib/gcforest/data/t2d/abundance_hmp-hmpii-t2d.txt'
    # abundance = 'data/t2d/abundance_hmp-hmpii-t2d.txt'
    f = pd.read_csv(abundance, sep='\t', header=None, index_col=0)
    f = f.T
    f = f.loc[(f != 0).any(axis=1)]
    f.set_index('sampleID', inplace=True)

    l = f['disease']
    l = l.replace(to_replace=['small_adenoma', 'large_adenoma'], value=['n', 'n'])
    label_encoder = LabelEncoder()
    integer_encoded = label_encoder.fit_transform(l)

    feature_identifier = 'k__'
    feat = [s for s in f.columns if sum([s2 in s for s2 in feature_identifier.split(':')]) > 0]
    f = f.loc[:, feat].astype('float')
    f = (f - f.min()) / (f.max() - f.min())

    return f, integer_encoded

def t2d_data():
    abundance = '../lib/gcforest/data/t2d/abundance_t2d_long-t2d_short.txt'
    # abundance = 'data/t2d/abundance_hmp-hmpii-t2d.txt'
    f = pd.read_csv(abundance, sep='\t', header=None, index_col=0)
    f = f.T
    f = f.loc[(f != 0).any(axis=1)]
    f.set_index('sampleID', inplace=True)

    l = f['disease']
    label_encoder = LabelEncoder()
    integer_encoded = label_encoder.fit_transform(l)

    feature_identifier = 'k__'
    feat = [s for s in f.columns if sum([s2 in s for s2 in feature_identifier.split(':')]) > 0]
    f = f.loc[:, feat].astype('float')
    f = (f - f.min()) / (f.max() - f.min())

    return f, integer_encoded

### OTU data
def colorectal_cancer_data():
    # Data reading
    otuinfile = '../lib/gcforest/data/colorectal/otu_colorectal_cancer'
    mapfile = '../lib/gcforest/data/colorectal/colorectal_metadata.tsv'
    disease_col = 'dx'

    data = pd.read_table(otuinfile, sep='\t', index_col=1)
    filtered_data = data.dropna(axis='columns', how='all')
    X = filtered_data.drop(['label', 'numOtus'], axis=1)
    X = (X - X.min()) / (X.max() - X.min())
    metadata = pd.read_table(mapfile, sep='\t', index_col=0)
    y = metadata[disease_col]
    ## Merge adenoma and normal in one-category called no-cancer, so we have binary classification
    y = y.replace(to_replace=['normal', 'adenoma'], value=['no-cancer', 'no-cancer'])

    encoder = LabelEncoder()
    integer_encoded = encoder.fit_transform(y)

    return X, integer_encoded


if __name__ == '__main__':
    col1 = cirrhosis_strain_data()
    print(len(col1))
