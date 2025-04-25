from sklearn.preprocessing import LabelEncoder, StandardScaler

def preprocess_data(df):
    df['Age'] = df['Age'].fillna(df['Age'].median())
    df['Fare'] = df['Fare'].fillna(df['Fare'].median())
    df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])
    df.drop(['Cabin', 'Name', 'Ticket'], axis=1, inplace=True)

    le = LabelEncoder()
    df['Sex'] = le.fit_transform(df['Sex'])
    df['Embarked'] = le.fit_transform(df['Embarked'])

    X = df.drop('Survived', axis=1)
    y = df['Survived']

    scaler = StandardScaler()
    X[['Age', 'Fare']] = scaler.fit_transform(X[['Age', 'Fare']])

    return X, y
