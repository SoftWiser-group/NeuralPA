[BugLab_Argument_Swapping]^if  ( f.shouldSkipField ( strategy )  )  {^37^^^^^35^42^if  ( strategy.shouldSkipField ( f )  )  {^[CLASS] DisjunctionExclusionStrategy  [METHOD] shouldSkipField [RETURN_TYPE] boolean   FieldAttributes f [VARIABLES] Collection  strategies  ExclusionStrategy  strategy  boolean  FieldAttributes  f  
[BugLab_Wrong_Literal]^return false;^38^^^^^35^42^return true;^[CLASS] DisjunctionExclusionStrategy  [METHOD] shouldSkipField [RETURN_TYPE] boolean   FieldAttributes f [VARIABLES] Collection  strategies  ExclusionStrategy  strategy  boolean  FieldAttributes  f  
[BugLab_Wrong_Literal]^return true;^41^^^^^35^42^return false;^[CLASS] DisjunctionExclusionStrategy  [METHOD] shouldSkipField [RETURN_TYPE] boolean   FieldAttributes f [VARIABLES] Collection  strategies  ExclusionStrategy  strategy  boolean  FieldAttributes  f  
[BugLab_Argument_Swapping]^if  ( clazz.shouldSkipClass ( strategy )  )  {^46^^^^^44^51^if  ( strategy.shouldSkipClass ( clazz )  )  {^[CLASS] DisjunctionExclusionStrategy  [METHOD] shouldSkipClass [RETURN_TYPE] boolean   Class<?> clazz [VARIABLES] Collection  strategies  Class  clazz  ExclusionStrategy  strategy  boolean  
[BugLab_Wrong_Literal]^return false;^47^^^^^44^51^return true;^[CLASS] DisjunctionExclusionStrategy  [METHOD] shouldSkipClass [RETURN_TYPE] boolean   Class<?> clazz [VARIABLES] Collection  strategies  Class  clazz  ExclusionStrategy  strategy  boolean  
[BugLab_Wrong_Literal]^return true;^50^^^^^44^51^return false;^[CLASS] DisjunctionExclusionStrategy  [METHOD] shouldSkipClass [RETURN_TYPE] boolean   Class<?> clazz [VARIABLES] Collection  strategies  Class  clazz  ExclusionStrategy  strategy  boolean  
