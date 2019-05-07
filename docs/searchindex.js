Search.setIndex({docnames:["env_wrappers","envs","estimators","index","policy_evaluation","policy_improvement","replay","utils"],envversion:{"sphinx.domains.c":1,"sphinx.domains.changeset":1,"sphinx.domains.cpp":1,"sphinx.domains.javascript":1,"sphinx.domains.math":2,"sphinx.domains.python":1,"sphinx.domains.rst":1,"sphinx.domains.std":1,"sphinx.ext.intersphinx":1,"sphinx.ext.todo":1,"sphinx.ext.viewcode":1,sphinx:56},filenames:["env_wrappers.rst","envs.rst","estimators.rst","index.rst","policy_evaluation.rst","policy_improvement.rst","replay.rst","utils.rst"],objects:{"wintermute.env_wrappers":{transformations:[0,0,0,"-"],wrappers:[0,0,0,"-"]},"wintermute.env_wrappers.transformations":{Downsample:[0,1,1,""],Normalize:[0,1,1,""],RBFFeaturize:[0,1,1,""],RGB2Y:[0,1,1,""],Standardize:[0,1,1,""]},"wintermute.env_wrappers.transformations.Downsample":{area:[0,2,1,""],cubic:[0,2,1,""],lanczos:[0,2,1,""],linear:[0,2,1,""],nearest:[0,2,1,""],transform:[0,3,1,""],update_env_specs:[0,3,1,""]},"wintermute.env_wrappers.transformations.Normalize":{transform:[0,3,1,""],update_env_specs:[0,3,1,""]},"wintermute.env_wrappers.transformations.RBFFeaturize":{transform:[0,3,1,""],update_env_specs:[0,3,1,""]},"wintermute.env_wrappers.transformations.RGB2Y":{transform:[0,3,1,""],update_env_specs:[0,3,1,""]},"wintermute.env_wrappers.transformations.Standardize":{transform:[0,3,1,""],update_env_specs:[0,3,1,""]},"wintermute.env_wrappers.wrappers":{DoneAfterLostLife:[0,1,1,""],FireResetEnv:[0,1,1,""],FrameStack:[0,1,1,""],MaxAndSkipEnv:[0,1,1,""],SqueezeRewards:[0,1,1,""],TorchWrapper:[0,1,1,""],TransformObservations:[0,1,1,""],get_wrapped_atari:[0,4,1,""]},"wintermute.env_wrappers.wrappers.DoneAfterLostLife":{reset:[0,3,1,""]},"wintermute.env_wrappers.wrappers.FireResetEnv":{reset:[0,3,1,""],step:[0,3,1,""]},"wintermute.env_wrappers.wrappers.FrameStack":{reset:[0,3,1,""],step:[0,3,1,""]},"wintermute.env_wrappers.wrappers.MaxAndSkipEnv":{reset:[0,3,1,""],step:[0,3,1,""]},"wintermute.env_wrappers.wrappers.SqueezeRewards":{reward:[0,3,1,""]},"wintermute.env_wrappers.wrappers.TorchWrapper":{observation:[0,3,1,""]},"wintermute.env_wrappers.wrappers.TransformObservations":{observation:[0,3,1,""]},"wintermute.envs":{ALE:[1,1,1,""],ale_env:[1,0,0,"-"]},"wintermute.envs.ALE":{close:[1,3,1,""],eval:[1,3,1,""],render:[1,3,1,""],reset:[1,3,1,""],step:[1,3,1,""],train:[1,3,1,""]},"wintermute.envs.ale_env":{ALE:[1,1,1,""]},"wintermute.envs.ale_env.ALE":{close:[1,3,1,""],eval:[1,3,1,""],render:[1,3,1,""],reset:[1,3,1,""],step:[1,3,1,""],train:[1,3,1,""]},"wintermute.estimators":{atari_net:[2,0,0,"-"],catch_net:[2,0,0,"-"]},"wintermute.estimators.atari_net":{AtariNet:[2,1,1,""],BootstrappedAtariNet:[2,1,1,""],get_feature_extractor:[2,4,1,""],get_head:[2,4,1,""]},"wintermute.estimators.atari_net.AtariNet":{feature_extractor:[2,2,1,""],forward:[2,3,1,""],head:[2,2,1,""],reset_parameters:[2,3,1,""]},"wintermute.estimators.atari_net.BootstrappedAtariNet":{forward:[2,3,1,""],parameters:[2,3,1,""],reset_parameters:[2,3,1,""]},"wintermute.estimators.catch_net":{CatchNet:[2,1,1,""]},"wintermute.estimators.catch_net.CatchNet":{forward:[2,3,1,""],get_attributes:[2,3,1,""]},"wintermute.policy_evaluation":{deterministic:[4,0,0,"-"],epsilon_greedy:[4,0,0,"-"],exploration_schedules:[4,0,0,"-"]},"wintermute.policy_evaluation.deterministic":{DeterministicOutput:[4,1,1,""],DeterministicPolicy:[4,1,1,""]},"wintermute.policy_evaluation.deterministic.DeterministicOutput":{action:[4,2,1,""],full:[4,2,1,""],q_value:[4,2,1,""]},"wintermute.policy_evaluation.deterministic.DeterministicPolicy":{cpu:[4,3,1,""],cuda:[4,3,1,""],get_action:[4,3,1,""],get_estimator_state:[4,3,1,""],set_estimator_state:[4,3,1,""]},"wintermute.policy_evaluation.epsilon_greedy":{EpsilonGreedyOutput:[4,1,1,""],EpsilonGreedyPolicy:[4,1,1,""]},"wintermute.policy_evaluation.epsilon_greedy.EpsilonGreedyOutput":{action:[4,2,1,""],full:[4,2,1,""],q_value:[4,2,1,""]},"wintermute.policy_evaluation.epsilon_greedy.EpsilonGreedyPolicy":{cpu:[4,3,1,""],cuda:[4,3,1,""],get_action:[4,3,1,""],get_estimator_state:[4,3,1,""],set_estimator_state:[4,3,1,""]},"wintermute.policy_evaluation.exploration_schedules":{constant_schedule:[4,4,1,""],float_range:[4,4,1,""],get_random_schedule:[4,4,1,""],get_schedule:[4,4,1,""],linear_schedule:[4,4,1,""],log_schedule:[4,4,1,""]},"wintermute.policy_improvement":{dqn_update:[5,0,0,"-"],optim_utils:[5,0,0,"-"]},"wintermute.policy_improvement.dqn_update":{DQNLoss:[5,1,1,""],DQNPolicyImprovement:[5,1,1,""],get_dqn_loss:[5,4,1,""],get_td_error:[5,4,1,""]},"wintermute.policy_improvement.dqn_update.DQNLoss":{loss:[5,2,1,""],q_targets:[5,2,1,""],q_values:[5,2,1,""]},"wintermute.policy_improvement.dqn_update.DQNPolicyImprovement":{get_estimator_state:[5,3,1,""],update_estimator:[5,3,1,""],update_target_estimator:[5,3,1,""]},"wintermute.policy_improvement.optim_utils":{float_range:[5,4,1,""],get_optimizer:[5,4,1,""],lr_schedule:[5,4,1,""]},"wintermute.replay":{ExperienceReplay:[6,1,1,""],MemoryEfficientExperienceReplay:[6,1,1,""],data_structures:[6,0,0,"-"],naive_experience_replay:[6,0,0,"-"],pinned_er:[6,0,0,"-"],prioritized_replay:[6,0,0,"-"],transitions:[6,0,0,"-"]},"wintermute.replay.MemoryEfficientExperienceReplay":{clear_ahead_results:[6,3,1,""],is_async:[6,2,1,""]},"wintermute.replay.data_structures":{PriorityQueue:[6,1,1,""],SumTree:[6,1,1,""]},"wintermute.replay.data_structures.PriorityQueue":{pop:[6,3,1,""],push:[6,3,1,""],update:[6,3,1,""]},"wintermute.replay.data_structures.SumTree":{get:[6,3,1,""],get_sum:[6,3,1,""],push:[6,3,1,""],update:[6,3,1,""]},"wintermute.replay.naive_experience_replay":{NaiveExperienceReplay:[6,1,1,""]},"wintermute.replay.naive_experience_replay.NaiveExperienceReplay":{push:[6,3,1,""]},"wintermute.replay.pinned_er":{PinnedExperienceReplay:[6,1,1,""]},"wintermute.replay.pinned_er.PinnedExperienceReplay":{is_async:[6,2,1,""]},"wintermute.replay.prioritized_replay":{ProportionalSampler:[6,1,1,""]},"wintermute.replay.prioritized_replay.ProportionalSampler":{batch_size:[6,2,1,""],update:[6,3,1,""]},"wintermute.replay.transitions":{ComparableTransition:[6,1,1,""],FullTransition:[6,1,1,""],HalfTransition:[6,1,1,""]},"wintermute.replay.transitions.ComparableTransition":{priority:[6,2,1,""],transition:[6,2,1,""]},"wintermute.replay.transitions.FullTransition":{action:[6,2,1,""],done:[6,2,1,""],meta:[6,2,1,""],next_state:[6,2,1,""],reward:[6,2,1,""],state:[6,2,1,""]},"wintermute.replay.transitions.HalfTransition":{action:[6,2,1,""],done:[6,2,1,""],meta:[6,2,1,""],reward:[6,2,1,""],state:[6,2,1,""]},"wintermute.utils":{torch_types:[7,0,0,"-"]},"wintermute.utils.torch_types":{TorchTypes:[7,1,1,""]},"wintermute.utils.torch_types.TorchTypes":{set_cuda:[7,3,1,""]},wintermute:{envs:[1,0,0,"-"],policy_improvement:[5,0,0,"-"],replay:[6,0,0,"-"],utils:[7,0,0,"-"]}},objnames:{"0":["py","module","Python module"],"1":["py","class","Python class"],"2":["py","attribute","Python attribute"],"3":["py","method","Python method"],"4":["py","function","Python function"]},objtypes:{"0":"py:module","1":"py:class","2":"py:attribute","3":"py:method","4":"py:function"},terms:{"100_000":6,"\u03b3":5,"boolean":0,"case":[0,5,6],"class":[0,1,2,4,5,6,7],"default":[0,2,4,5,6],"float":[0,4,5,6],"function":[2,5,6],"int":[0,2,4,6],"return":[0,1,2,4,5,6],"super":3,"true":[0,1,2,5,6],"try":6,"while":2,ALE:1,The:[0,4,5,6],These:0,__call__:5,a_t:5,abov:6,abstracttransform:0,accept:0,accumul:5,across:5,action:[0,1,4,5,6],action_spac:4,actual:6,adam:5,add:6,advantag:6,after:[0,1],afterward:2,agent:0,algorithm:0,alia:[4,5,6],all:[1,2,6],alloc:6,alpha:5,also:0,although:2,alwai:5,amount:0,ani:6,api:3,appli:0,approxim:0,arcad:1,architectur:2,area:0,arg:[2,4],argument:6,arxiv:6,assum:5,async_memori:6,asynchron:6,asyncron:6,atari:[0,2],atari_pi:1,atarinet:2,attribut:6,auxiliari:0,avail:[0,5,6],base:[0,1,2,4,5,6,7],baselin:0,batch:[0,5,6],batch_siz:6,begin:5,behaviour:0,below:5,best:4,bia:2,binari:6,bind:1,blob:1,bool:[5,6],boot_no:2,bootstrap_arg:6,bootstrappedatarinet:2,britz:0,buffer:6,build:6,byproduct:5,call:[0,2,5],callback:5,can:[2,6],cancel:6,capac:6,care:2,catchnet:2,chang:0,circular:6,clear_ahead_result:6,clip:1,clip_rewards_v:1,clone:5,close:1,closer:1,collat:6,com:[1,3],compar:6,comparabletransit:6,compon:2,compris:6,comput:[2,5],concaten:6,concatent:1,configur:2,constant:4,constant_schedul:4,construct:5,contain:[0,5,6],content:6,convert:0,core:0,corner:6,cpu:4,credit:1,cubic:0,cuda:[4,6,7],current:[0,6],custom:5,cyclic:6,data_structur:6,debug:0,deep:5,deepmind:0,defin:2,degrad:4,delta:5,delta_i:5,denni:0,depend:5,deterministicoutput:4,deterministicpolici:4,devic:[1,5,6],diagnost:0,dict:[0,6],differ:5,dimension:2,discount:5,docstr:6,doing:5,don:6,done:[0,6],done_mask:5,doneafterlostlif:0,doubl:5,downsampl:0,dqn:[1,6],dqn_updat:5,dqnloss:5,dqnpolicyimprov:5,duh:6,duper:3,dure:[1,4,5],dynam:0,each:[0,2],easi:3,either:[5,6],element:[5,6],enabl:6,end:[0,1,4,5],ensembl:2,env:[0,5],env_nam:0,env_wrapp:0,environ:[1,3],episod:0,eps:5,epsilon:4,epsilongreedyoutput:4,epsilongreedypolici:4,error:5,estim:[3,4,5],estimator_st:4,etc:6,eval:1,evalu:[0,3],eventu:6,everi:[0,2],exampl:[0,5],expect:6,expens:6,experi:5,experience_replai:5,experiencereplai:6,explor:4,extract:0,extractor:2,factor:5,factori:6,fals:[0,2,4,5,6,7],faster:6,featur:[0,2],feature_extractor:2,field:[4,5,6],file:6,fireresetenv:0,first:1,flatexperiencereplai:6,float_rang:[4,5],floringogianu:3,form:6,former:2,forward:2,found:6,four:6,frame:[0,1,6],framestack:0,from:[0,6],full:[2,4,6],full_transit:6,fulltransit:6,further:0,game:[0,1,2],gamma:5,get:6,get_act:4,get_attribut:2,get_dqn_loss:5,get_estimator_st:[4,5],get_feature_extractor:2,get_head:2,get_optim:5,get_random_schedul:4,get_schedul:4,get_sum:6,get_td_error:5,get_wrapped_atari:0,git:3,github:[1,3],given:[0,4,6],gradient:5,greedi:4,group:2,gym:[0,1],half:6,halftransit:6,happen:[5,6],has:0,head:2,heap:6,height:0,help:0,helper:6,hidden_s:2,hist:0,hist_len:[2,6],histori:6,history_len:1,history_length:1,hold:6,hook:2,howev:5,http:[1,6],huber:5,idx:6,ignor:2,imag:0,implement:6,improv:[3,6],indic:2,infer:2,info:0,inform:0,initi:0,input:2,input_ch:2,input_channel:2,input_depth:2,insert:6,instal:3,instanc:2,instead:2,interpol:0,interv:6,is_async:6,is_doubl:5,is_train:4,item:6,iter:[2,4],its:[4,5,6],kaixhin:1,keep:[2,6],kernel:0,kwarg:[0,6],lanczo:0,largest:6,last:[0,6],latter:2,layer:2,lcon:0,leaf:6,learn:[0,1,5],length:6,librari:[3,5],life:[0,1],lift:0,linear:[0,4],linear_schedul:4,list:[5,6],log_schedul:4,logarithm:4,loss:5,loss_fn:5,lost:0,low:2,lr_schedul:5,lumin:0,make:6,mask_dtyp:6,master:1,max:[0,5],max_episode_length:1,maxandskipenv:0,mean:[0,2],mem_efficient_experience_replai:6,member:[0,6],memoryefficientexperiencereplai:6,meta:6,method:5,mid:2,mode:[0,1,2],model:[2,5],modul:5,more:1,most:0,mseloss:5,multipl:0,n_compon:0,naive_experience_replai:6,naiveexperiencereplai:6,name:[4,5],namespac:5,nearest:0,need:2,net:5,network:[2,5],neural:2,new_prior:6,next:6,next_stat:[5,6],no_gym:0,node:6,none:[2,5,6],normal:0,notebook:0,number:[4,5,6],numpi:0,object:[0,1,4,5,6,7],obs:0,observ:6,observationwrapp:0,one:[0,1,2,5],onli:[0,6],onlin:5,openai:[0,1],oppos:6,optim:2,optim_step:6,optim_util:5,optimi:6,option:[0,4,5,6],order:6,org:6,origin:1,other:[0,5],otherwis:[2,5],out_siz:2,output:[2,4],over:[0,1],overrid:5,overridden:2,overwritten:0,paper:1,paramet:[0,2,5,6],pass:2,pdf:6,percol:6,perform:2,pinned_:6,pinnedexperiencereplai:6,pip:3,polici:[3,6],policy_improv:5,pop:6,predict:2,preprocess:0,previou:0,priorit:5,prioriti:6,prioritized_replai:6,priorityqueu:6,prob:4,probabl:1,proport:6,proportionalsampl:6,proto:2,provid:[0,1,2],push:[5,6],put:5,python:1,pytorch:5,q_target:5,q_target_valu:5,q_valu:[4,5],queue:6,r_t:5,rainbow:1,ram:6,rang:[5,6],rank:6,ranksampl:6,rbf:0,rbffeatur:0,reach:0,recip:2,recurs:2,reduct:5,refer:5,regist:2,reinitializez:2,remain:4,render:1,repair:6,repeat:0,replai:[3,5],request:6,reset:[0,1],reset_paramet:2,respons:0,result:0,reward:[0,1,5,6],rewardwrapp:0,rgb2y:0,rgb:0,rule:5,run:[0,2],s_t:5,same:5,sampl:[0,5,6],scale:0,schedul:4,screen_siz:6,scren_dtyp:6,seed:1,select:4,separ:2,set:1,set_cuda:7,set_estimator_st:4,sever:[5,6],shared_bia:2,shoplift:0,should:[2,6],sign:0,silent:2,simpl:5,sinc:[2,5],singl:5,size:[5,6],skip:0,smoothl1loss:5,some:0,sometim:0,sourc:[0,1,2,4,5,6,7],space:0,specif:0,squeezereward:0,stack:0,standard:0,start:[4,5,6],state:[0,4,5,6],statist:2,step:[0,1,2,4,5,6],steps_no:[4,5],sticki:1,sticky_action_p:1,store:6,str:4,strategi:4,strip:6,subclass:2,subsequ:4,subtree_sum:6,sum:[0,6],sumtre:6,support:6,take:[2,4,6],target:[0,5],target_estim:5,target_update_freq:5,tempor:5,tensor:[2,5,6],term:6,text:5,them:[0,2],therefor:5,thi:[0,1,2,5,6],thing:[5,6],thread:6,through:2,time:6,timestep:0,torch:[0,2,5,6],torchtyp:7,torchwrapp:0,toward:4,tradition:0,train:[0,1,2],train_step:5,transformobserv:0,transit:5,tree:6,tupl:[0,4,5,6],type:[0,1,2,4,5,6],uint8:6,undefin:0,uniform:2,union:6,unit:0,updat:[0,6],update_env_spec:0,update_estim:5,update_target_estim:5,usag:5,use:5,use_cuda:7,used:[0,2,6],useful:5,uses:6,using:[0,5,6],usual:6,valu:[4,5,6],varianc:0,variant:5,variou:[4,6],verbos:0,version:6,wait:6,want:5,warmup:4,warmup_step:4,when:[0,2,5],whether:[0,5],which:[0,4,6],width:0,wih:6,wintermut:[0,5],within:2,without:0,work:0,wrap:0,wrapper:[1,3],xavier:2,you:0,zero:0},titles:["Environment Wrappers","wintermute.envs package","Estimators","Wintermute","Policy Evaluation","Policy Improvement","Replay","wintermute.utils package"],titleterms:{ale_env:1,atari_net:2,catch_net:2,content:[1,7],data:6,determinist:4,dqn:5,effici:6,env:1,environ:0,epsilon_greedi:4,estim:2,evalu:4,experi:6,exploration_schedul:4,improv:5,memori:6,modul:[1,2,4,6,7],naiv:6,observ:0,optim:5,packag:[1,7],pin:6,polici:[4,5],policy_evalu:4,priorit:6,replai:6,structur:6,submodul:[1,7],torch_typ:7,transform:0,transit:6,updat:5,util:[5,7],wintermut:[1,2,3,4,6,7],wrapper:0}})