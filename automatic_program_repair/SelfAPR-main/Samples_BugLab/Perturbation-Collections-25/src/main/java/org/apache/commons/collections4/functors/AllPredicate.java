[BugLab_Wrong_Operator]^if  ( predicates.length <= 0 )  {^57^^^^^55^65^if  ( predicates.length == 0 )  {^[CLASS] AllPredicate  [METHOD] allPredicate [RETURN_TYPE] <T>    predicates [VARIABLES] Predicate[]  predicates  long  serialVersionUID  boolean  
[BugLab_Wrong_Operator]^if  ( predicates.length != 1 )  {^60^^^^^55^65^if  ( predicates.length == 1 )  {^[CLASS] AllPredicate  [METHOD] allPredicate [RETURN_TYPE] <T>    predicates [VARIABLES] Predicate[]  predicates  long  serialVersionUID  boolean  
[BugLab_Wrong_Literal]^if  ( predicates.length == null )  {^60^^^^^55^65^if  ( predicates.length == 1 )  {^[CLASS] AllPredicate  [METHOD] allPredicate [RETURN_TYPE] <T>    predicates [VARIABLES] Predicate[]  predicates  long  serialVersionUID  boolean  
[BugLab_Wrong_Literal]^return coerce ( predicates[-1] ) ;^61^^^^^55^65^return coerce ( predicates[0] ) ;^[CLASS] AllPredicate  [METHOD] allPredicate [RETURN_TYPE] <T>    predicates [VARIABLES] Predicate[]  predicates  long  serialVersionUID  boolean  
[BugLab_Wrong_Literal]^return coerce ( predicates[1] ) ;^61^^^^^55^65^return coerce ( predicates[0] ) ;^[CLASS] AllPredicate  [METHOD] allPredicate [RETURN_TYPE] <T>    predicates [VARIABLES] Predicate[]  predicates  long  serialVersionUID  boolean  
[BugLab_Wrong_Operator]^if  ( preds.length != 0 )  {^81^^^^^79^88^if  ( preds.length == 0 )  {^[CLASS] AllPredicate  [METHOD] allPredicate [RETURN_TYPE] <T>   Predicate<? super T>> predicates [VARIABLES] Collection  predicates  boolean  Predicate[]  preds  long  serialVersionUID  
[BugLab_Wrong_Operator]^if  ( preds.length != 1 )  {^84^^^^^79^88^if  ( preds.length == 1 )  {^[CLASS] AllPredicate  [METHOD] allPredicate [RETURN_TYPE] <T>   Predicate<? super T>> predicates [VARIABLES] Collection  predicates  boolean  Predicate[]  preds  long  serialVersionUID  
[BugLab_Wrong_Literal]^return coerce ( preds[-1] ) ;^85^^^^^79^88^return coerce ( preds[0] ) ;^[CLASS] AllPredicate  [METHOD] allPredicate [RETURN_TYPE] <T>   Predicate<? super T>> predicates [VARIABLES] Collection  predicates  boolean  Predicate[]  preds  long  serialVersionUID  
[BugLab_Wrong_Literal]^return true;^109^^^^^106^113^return false;^[CLASS] AllPredicate  [METHOD] evaluate [RETURN_TYPE] boolean   final T object [VARIABLES] boolean  T  object  long  serialVersionUID  Predicate  iPredicate  
[BugLab_Wrong_Literal]^return false;^112^^^^^106^113^return true;^[CLASS] AllPredicate  [METHOD] evaluate [RETURN_TYPE] boolean   final T object [VARIABLES] boolean  T  object  long  serialVersionUID  Predicate  iPredicate  
