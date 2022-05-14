!pip3 uninstall twint -y
!pip3 install --user --upgrade git+https://github.com/twintproject/twint.git@origin/master#egg=twint
import twint
import pandas as pd
# Configure
import nest_asyncio

nest_asyncio.apply()

c = twint.Config()
c.Search = "#abortionismurder OR #antiabortion OR #stopabortion OR #noabortion OR #saynotoabortion"
c.Lang = "en"

c.Limit = 500
c.Pandas = True
# Run
twint.run.Search(c)

df = twint.storage.panda.Tweets_df
df = df[["id","tweet","hashtags"]]
df["Type"] = "Abortion"
df

nest_asyncio.apply()

c4 = twint.Config()
c4.Search = "#abortionismurder OR #antiabortion OR #stopabortion OR #noabortion OR #saynotoabortion AND jesus"
c4.Lang = "en"

c4.Limit = 500
c4.Pandas = True
# Run
twint.run.Search(c)

df4 = twint.storage.panda.Tweets_df
df4 = df[["id","tweet","hashtags"]]
df4["Type"] = "Abortion"
df4

c1 = twint.Config()
c1.Search = "PCOS AND overweight OR obese OR fertility"
c1.Lang = "en"

c1.Limit = 500
c1.Pandas = True
# Run
twint.run.Search(c1)

df1 = twint.storage.panda.Tweets_df
df1 = df1[["id","tweet","hashtags"]]
df1["Type"] = "PCOS"
df1

c2 = twint.Config()
c2.Search = "#VaccineSideEffects women"
c2.Lang = "en"

c2.Limit = 500
c2.Pandas = True
# Run
twint.run.Search(c2)

df2 = twint.storage.panda.Tweets_df
df2 = df2[["id","tweet","hashtags"]]
df2["Type"] = "Vaccine"
df2

c3 = twint.Config()
c3.Search = "periods OR dirty blood OR menstruation"
c3.Lang = "en"

c3.Limit = 500
c3.Pandas = True
# Run
twint.run.Search(c3)

df3 = twint.storage.panda.Tweets_df
df3 = df3[["id","tweet","hashtags"]]
df3["Type"] = "Menstruation"
df3

df_final = pd.concat([df, df1, df2, df3, df4])
df_final.to_csv('test1.csv',index=False)

import pandas as pd
final_list = []

submission = reddit.submission(url="https://www.reddit.com/r/AskReddit/comments/q2mbmm/do_you_believe_abortion_should_be_legal_why_or/?sort=controversial")
submission.comments.replace_more(limit=0)
for comment in submission.comments.list():
    final_list.append([comment.id,comment.body,"Abortion"])
df_new = pd.DataFrame(final_list,columns=['id','Text','Type'])
df_new

import pandas as pd
final_list = []

submission = reddit.submission(url="https://www.reddit.com/r/confidentlyincorrect/comments/sw1163/no_first_they_said_it_changes_your_dna_then_they/")
submission.comments.replace_more(limit=0)
for comment in submission.comments.list():
    final_list.append([comment.id,comment.body,"Vaccine"])
df_1 = pd.DataFrame(final_list,columns=['id','Text','Type'])
df_1

import pandas as pd
final_list = []

submission = reddit.submission(url="https://www.reddit.com/r/fatlogic/comments/3mqxbi/in_fact_two_thirds_of_women_with_pcos_are/?sort=controversial")
submission.comments.replace_more(limit=0)
for comment in submission.comments.list():
    final_list.append([comment.id,comment.body,"PCOS"])
df_2 = pd.DataFrame(final_list,columns=['id','Text','Type'])
df_2
