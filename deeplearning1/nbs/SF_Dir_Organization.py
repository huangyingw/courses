
# coding: utf-8

# In[1]:


# run this script from the same dir that your test and train data live in

import os
import glob
import random
import pandas as pd
import shutil

# point to your training images
train_dir = '/home/ubuntu/nbs/data/StateFarm/train'

# point to the 'driver_imgs_list.csv'
lookup = '/home/ubuntu/nbs/data/StateFarm/driver_imgs_list.csv'

# point to the validation directory, which will be created in the next block
val_dir = '/home/ubuntu/nbs/data/StateFarm/valid'


# In[23]:


# creates a directory, called 'validation', and into it, creates 10
# subdirs, one for each class.

directory = 'valid'
if not os.path.exists(directory):
    os.makedirs(directory)
cwd = os.getcwd()
# path = cwd + '/' + directory
path = val_dir
[os.mkdir('{}/c{}'.format(path, i)) for i in range(10)]


# In[2]:


# read in the 'driver_imgs_list.csv' file
df = pd.read_csv(lookup)


# In[18]:


# There are 26 unique drivers in the train set.
# We want 20%, so pick 5 at random
driver_names = [
    'p002',
    'p012',
    'p014',
    'p015',
    'p016',
    'p021',
    'p022',
    'p024',
    'p026',
    'p035',
    'p039',
    'p041',
    'p042',
    'p045',
    'p047',
    'p049',
    'p050',
    'p051',
    'p052',
    'p056',
    'p061',
    'p064',
    'p066',
    'p072',
    'p075',
    'p081']
driver_names_for_validation = random.sample(driver_names, 5)


# In[19]:


# filter df by selecting rows with the subjects we want
df = df[df['subject'].isin(driver_names_for_validation)]


# In[29]:


#dftest = df.loc[0:1]
#    print str(i) + ': ' +  'moving ' + to_move

# do the move here:
i = 0
for index, row in df.iterrows():
    to_move = train_dir + '/' + row['classname'] + '/' + row['img']
    print to_move
    i = i + 1
    move_to = val_dir + '/' + row['classname']
    print move_to
    shutil.move(to_move, move_to)


# In[30]:


print 'files to move: ', df.shape[0]
print 'files moved:   ', i
