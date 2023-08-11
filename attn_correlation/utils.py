import pandas as pd
import os


class EyeTrackingDataLoader:

    def __init__(self, data_dir: str):
        self.data_dir = data_dir

    def __load_and_merge_users_dfs(self) -> pd.DataFrame:
        users_dfs = []
        for user_file_name in os.listdir(self.data_dir):
            src_path = os.path.join(self.data_dir, user_file_name)
            user_df = pd.read_csv(src_path)[['trialid', 'sentnum', 'ianum', 'ia']]
            users_dfs.append(user_df)
        merged_df = pd.concat(users_dfs, ignore_index=True).drop_duplicates()
        return merged_df

    def load_sentences(self) -> pd.DataFrame:
        """
        This method creates a DataFrame with the following columns:
        - sent_id: containing a unique key for the sentence computed as the concatenation of 'trialid' and 'sentnum'
        - sentence: contains the list of words of a sentence, sorted by 'ianum'
        A bit of processing is necessary since the sentences have been split in pieces randomly located on users files.
        Moreover, not all users read all the sentences, and all pieces of it.
        """
        merged_user_df = self.__load_and_merge_users_dfs()
        sentences_df = merged_user_df.sort_values(by=['trialid', 'sentnum', 'ianum']).groupby(['trialid', 'sentnum'])[
            'ia'].apply(list).reset_index()
        sent_id_column = sentences_df['trialid'].astype(int).astype(str) + '_' + sentences_df['sentnum'].astype(
            int).astype(str)
        sentences_df.insert(0, 'sent_id', sent_id_column)  # I wanted it in first position :)
        sentences_df = sentences_df.drop(['trialid', 'sentnum'], axis=1)
        sentences_df.rename(columns={'ia': 'sentence'}, inplace=True)
        return sentences_df
