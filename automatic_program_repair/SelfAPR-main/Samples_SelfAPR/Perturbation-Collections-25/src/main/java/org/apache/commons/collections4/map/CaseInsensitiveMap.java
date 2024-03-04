[P1_Replace_Type]^private static final  int  serialVersionUID = -7074655917369299456L;^66^^^^^61^71^private static final long serialVersionUID = -7074655917369299456L;^[CLASS] CaseInsensitiveMap   [VARIABLES] 
[P8_Replace_Mix]^private static final long serialVersionUID = -7074655917369299456;^66^^^^^61^71^private static final long serialVersionUID = -7074655917369299456L;^[CLASS] CaseInsensitiveMap   [VARIABLES] 
[P14_Delete_Statement]^^72^^^^^71^73^super ( DEFAULT_CAPACITY, DEFAULT_LOAD_FACTOR, DEFAULT_THRESHOLD ) ;^[CLASS] CaseInsensitiveMap  [METHOD] <init> [RETURN_TYPE] CaseInsensitiveMap()   [VARIABLES] long  serialVersionUID  boolean  
[P14_Delete_Statement]^^82^^^^^81^83^super ( initialCapacity ) ;^[CLASS] CaseInsensitiveMap  [METHOD] <init> [RETURN_TYPE] CaseInsensitiveMap(int)   final int initialCapacity [VARIABLES] long  serialVersionUID  int  initialCapacity  boolean  
[P11_Insert_Donor_Statement]^super ( map ) ;super ( initialCapacity ) ;^82^^^^^81^83^super ( initialCapacity ) ;^[CLASS] CaseInsensitiveMap  [METHOD] <init> [RETURN_TYPE] CaseInsensitiveMap(int)   final int initialCapacity [VARIABLES] long  serialVersionUID  int  initialCapacity  boolean  
[P11_Insert_Donor_Statement]^super ( initialCapacity, loadFactor ) ;super ( initialCapacity ) ;^82^^^^^81^83^super ( initialCapacity ) ;^[CLASS] CaseInsensitiveMap  [METHOD] <init> [RETURN_TYPE] CaseInsensitiveMap(int)   final int initialCapacity [VARIABLES] long  serialVersionUID  int  initialCapacity  boolean  
[P5_Replace_Variable]^super (  loadFactor ) ;^95^^^^^94^96^super ( initialCapacity, loadFactor ) ;^[CLASS] CaseInsensitiveMap  [METHOD] <init> [RETURN_TYPE] CaseInsensitiveMap(int,float)   final int initialCapacity final float loadFactor [VARIABLES] boolean  float  loadFactor  long  serialVersionUID  int  initialCapacity  
[P5_Replace_Variable]^super ( initialCapacity ) ;^95^^^^^94^96^super ( initialCapacity, loadFactor ) ;^[CLASS] CaseInsensitiveMap  [METHOD] <init> [RETURN_TYPE] CaseInsensitiveMap(int,float)   final int initialCapacity final float loadFactor [VARIABLES] boolean  float  loadFactor  long  serialVersionUID  int  initialCapacity  
[P5_Replace_Variable]^super ( loadFactor, initialCapacity ) ;^95^^^^^94^96^super ( initialCapacity, loadFactor ) ;^[CLASS] CaseInsensitiveMap  [METHOD] <init> [RETURN_TYPE] CaseInsensitiveMap(int,float)   final int initialCapacity final float loadFactor [VARIABLES] boolean  float  loadFactor  long  serialVersionUID  int  initialCapacity  
[P14_Delete_Statement]^^95^^^^^94^96^super ( initialCapacity, loadFactor ) ;^[CLASS] CaseInsensitiveMap  [METHOD] <init> [RETURN_TYPE] CaseInsensitiveMap(int,float)   final int initialCapacity final float loadFactor [VARIABLES] boolean  float  loadFactor  long  serialVersionUID  int  initialCapacity  
[P11_Insert_Donor_Statement]^super ( initialCapacity ) ;super ( initialCapacity, loadFactor ) ;^95^^^^^94^96^super ( initialCapacity, loadFactor ) ;^[CLASS] CaseInsensitiveMap  [METHOD] <init> [RETURN_TYPE] CaseInsensitiveMap(int,float)   final int initialCapacity final float loadFactor [VARIABLES] boolean  float  loadFactor  long  serialVersionUID  int  initialCapacity  
[P14_Delete_Statement]^^109^^^^^108^110^super ( map ) ;^[CLASS] CaseInsensitiveMap  [METHOD] <init> [RETURN_TYPE] Map)   Map<? extends K, ? extends V> map [VARIABLES] Map  map  long  serialVersionUID  boolean  
[P11_Insert_Donor_Statement]^super ( initialCapacity ) ;super ( map ) ;^109^^^^^108^110^super ( map ) ;^[CLASS] CaseInsensitiveMap  [METHOD] <init> [RETURN_TYPE] Map)   Map<? extends K, ? extends V> map [VARIABLES] Map  map  long  serialVersionUID  boolean  
[P2_Replace_Operator]^if  ( key == null )  {^124^^^^^123^132^if  ( key != null )  {^[CLASS] CaseInsensitiveMap  [METHOD] convertKey [RETURN_TYPE] Object   Object key [VARIABLES] Object  key  boolean  long  serialVersionUID  int  i  char[]  chars  
[P6_Replace_Expression]^if  ( chars.length - 1 )  {^124^^^^^123^132^if  ( key != null )  {^[CLASS] CaseInsensitiveMap  [METHOD] convertKey [RETURN_TYPE] Object   Object key [VARIABLES] Object  key  boolean  long  serialVersionUID  int  i  char[]  chars  
[P6_Replace_Expression]^if  ( i >= 0 )  {^124^^^^^123^132^if  ( key != null )  {^[CLASS] CaseInsensitiveMap  [METHOD] convertKey [RETURN_TYPE] Object   Object key [VARIABLES] Object  key  boolean  long  serialVersionUID  int  i  char[]  chars  
[P8_Replace_Mix]^if  ( key != this )  {^124^^^^^123^132^if  ( key != null )  {^[CLASS] CaseInsensitiveMap  [METHOD] convertKey [RETURN_TYPE] Object   Object key [VARIABLES] Object  key  boolean  long  serialVersionUID  int  i  char[]  chars  
[P15_Unwrap_Block]^final char[] chars = key.toString().toCharArray(); for (int i = (chars.length) - 1; i >= 0; i--) {    chars[i] = java.lang.Character.toLowerCase(java.lang.Character.toUpperCase(chars[i]));}; return new java.lang.String(chars);^124^125^126^127^128^123^132^if  ( key != null )  { final char[] chars = key.toString (  ) .toCharArray (  ) ; for  ( int i = chars.length - 1; i >= 0; i-- )  { chars[i] = Character.toLowerCase ( Character.toUpperCase ( chars[i] )  ) ; }^[CLASS] CaseInsensitiveMap  [METHOD] convertKey [RETURN_TYPE] Object   Object key [VARIABLES] Object  key  boolean  long  serialVersionUID  int  i  char[]  chars  
[P16_Remove_Block]^^124^125^126^127^128^123^132^if  ( key != null )  { final char[] chars = key.toString (  ) .toCharArray (  ) ; for  ( int i = chars.length - 1; i >= 0; i-- )  { chars[i] = Character.toLowerCase ( Character.toUpperCase ( chars[i] )  ) ; }^[CLASS] CaseInsensitiveMap  [METHOD] convertKey [RETURN_TYPE] Object   Object key [VARIABLES] Object  key  boolean  long  serialVersionUID  int  i  char[]  chars  
[P1_Replace_Type]^for  (  long  i = chars.length - 1; i >= 0; i-- )  {^126^^^^^123^132^for  ( int i = chars.length - 1; i >= 0; i-- )  {^[CLASS] CaseInsensitiveMap  [METHOD] convertKey [RETURN_TYPE] Object   Object key [VARIABLES] Object  key  boolean  long  serialVersionUID  int  i  char[]  chars  
[P2_Replace_Operator]^for  ( int i = chars.length  >  1; i >= 0; i-- )  {^126^^^^^123^132^for  ( int i = chars.length - 1; i >= 0; i-- )  {^[CLASS] CaseInsensitiveMap  [METHOD] convertKey [RETURN_TYPE] Object   Object key [VARIABLES] Object  key  boolean  long  serialVersionUID  int  i  char[]  chars  
[P2_Replace_Operator]^for  ( int i = chars.length - 1; i > 0; i-- )  {^126^^^^^123^132^for  ( int i = chars.length - 1; i >= 0; i-- )  {^[CLASS] CaseInsensitiveMap  [METHOD] convertKey [RETURN_TYPE] Object   Object key [VARIABLES] Object  key  boolean  long  serialVersionUID  int  i  char[]  chars  
[P3_Replace_Literal]^for  ( int i = chars.length ; i >= 0; i-- )  {^126^^^^^123^132^for  ( int i = chars.length - 1; i >= 0; i-- )  {^[CLASS] CaseInsensitiveMap  [METHOD] convertKey [RETURN_TYPE] Object   Object key [VARIABLES] Object  key  boolean  long  serialVersionUID  int  i  char[]  chars  
[P3_Replace_Literal]^for  ( int i = chars.length - 1; i >= i; i-- )  {^126^^^^^123^132^for  ( int i = chars.length - 1; i >= 0; i-- )  {^[CLASS] CaseInsensitiveMap  [METHOD] convertKey [RETURN_TYPE] Object   Object key [VARIABLES] Object  key  boolean  long  serialVersionUID  int  i  char[]  chars  
[P5_Replace_Variable]^for  ( int i = chars.length.length - 1; i >= 0; i-- )  {^126^^^^^123^132^for  ( int i = chars.length - 1; i >= 0; i-- )  {^[CLASS] CaseInsensitiveMap  [METHOD] convertKey [RETURN_TYPE] Object   Object key [VARIABLES] Object  key  boolean  long  serialVersionUID  int  i  char[]  chars  
[P5_Replace_Variable]^for  ( chars.lengthnt i = i - 1; i >= 0; i-- )  {^126^^^^^123^132^for  ( int i = chars.length - 1; i >= 0; i-- )  {^[CLASS] CaseInsensitiveMap  [METHOD] convertKey [RETURN_TYPE] Object   Object key [VARIABLES] Object  key  boolean  long  serialVersionUID  int  i  char[]  chars  
[P5_Replace_Variable]^for  ( int i = chars - 1; i >= 0; i-- )  {^126^^^^^123^132^for  ( int i = chars.length - 1; i >= 0; i-- )  {^[CLASS] CaseInsensitiveMap  [METHOD] convertKey [RETURN_TYPE] Object   Object key [VARIABLES] Object  key  boolean  long  serialVersionUID  int  i  char[]  chars  
[P14_Delete_Statement]^^127^^^^^123^132^chars[i] = Character.toLowerCase ( Character.toUpperCase ( chars[i] )  ) ;^[CLASS] CaseInsensitiveMap  [METHOD] convertKey [RETURN_TYPE] Object   Object key [VARIABLES] Object  key  boolean  long  serialVersionUID  int  i  char[]  chars  
[P1_Replace_Type]^return new char ( chars ) ;^129^^^^^123^132^return new String ( chars ) ;^[CLASS] CaseInsensitiveMap  [METHOD] convertKey [RETURN_TYPE] Object   Object key [VARIABLES] Object  key  boolean  long  serialVersionUID  int  i  char[]  chars  
[P7_Replace_Invocation]^final char[] chars = key.toString (  ) .String (  ) ;^125^^^^^123^132^final char[] chars = key.toString (  ) .toCharArray (  ) ;^[CLASS] CaseInsensitiveMap  [METHOD] convertKey [RETURN_TYPE] Object   Object key [VARIABLES] Object  key  boolean  long  serialVersionUID  int  i  char[]  chars  
[P8_Replace_Mix]^final char[] chars = key.toString (  )  .String ( chars )  ;^125^^^^^123^132^final char[] chars = key.toString (  ) .toCharArray (  ) ;^[CLASS] CaseInsensitiveMap  [METHOD] convertKey [RETURN_TYPE] Object   Object key [VARIABLES] Object  key  boolean  long  serialVersionUID  int  i  char[]  chars  
[P1_Replace_Type]^for  (  short  i = chars.length - 1; i >= 0; i-- )  {^126^^^^^123^132^for  ( int i = chars.length - 1; i >= 0; i-- )  {^[CLASS] CaseInsensitiveMap  [METHOD] convertKey [RETURN_TYPE] Object   Object key [VARIABLES] Object  key  boolean  long  serialVersionUID  int  i  char[]  chars  
[P2_Replace_Operator]^for  ( int i = chars.length  ^  1; i >= 0; i-- )  {^126^^^^^123^132^for  ( int i = chars.length - 1; i >= 0; i-- )  {^[CLASS] CaseInsensitiveMap  [METHOD] convertKey [RETURN_TYPE] Object   Object key [VARIABLES] Object  key  boolean  long  serialVersionUID  int  i  char[]  chars  
[P3_Replace_Literal]^for  ( int i = chars.length - i; i >= 0; i-- )  {^126^^^^^123^132^for  ( int i = chars.length - 1; i >= 0; i-- )  {^[CLASS] CaseInsensitiveMap  [METHOD] convertKey [RETURN_TYPE] Object   Object key [VARIABLES] Object  key  boolean  long  serialVersionUID  int  i  char[]  chars  
[P14_Delete_Statement]^^125^^^^^123^132^final char[] chars = key.toString (  ) .toCharArray (  ) ;^[CLASS] CaseInsensitiveMap  [METHOD] convertKey [RETURN_TYPE] Object   Object key [VARIABLES] Object  key  boolean  long  serialVersionUID  int  i  char[]  chars  
[P8_Replace_Mix]^final char[] chars = key .Object (  )  .toCharArray (  ) ;^125^^^^^123^132^final char[] chars = key.toString (  ) .toCharArray (  ) ;^[CLASS] CaseInsensitiveMap  [METHOD] convertKey [RETURN_TYPE] Object   Object key [VARIABLES] Object  key  boolean  long  serialVersionUID  int  i  char[]  chars  
[P2_Replace_Operator]^for  ( int i = chars.length  >=  1; i >= 0; i-- )  {^126^^^^^123^132^for  ( int i = chars.length - 1; i >= 0; i-- )  {^[CLASS] CaseInsensitiveMap  [METHOD] convertKey [RETURN_TYPE] Object   Object key [VARIABLES] Object  key  boolean  long  serialVersionUID  int  i  char[]  chars  
[P2_Replace_Operator]^for  ( int i = chars.length - 1; i == 0; i-- )  {^126^^^^^123^132^for  ( int i = chars.length - 1; i >= 0; i-- )  {^[CLASS] CaseInsensitiveMap  [METHOD] convertKey [RETURN_TYPE] Object   Object key [VARIABLES] Object  key  boolean  long  serialVersionUID  int  i  char[]  chars  
[P3_Replace_Literal]^for  ( int i = chars.length - 1; i >= -1; i-- )  {^126^^^^^123^132^for  ( int i = chars.length - 1; i >= 0; i-- )  {^[CLASS] CaseInsensitiveMap  [METHOD] convertKey [RETURN_TYPE] Object   Object key [VARIABLES] Object  key  boolean  long  serialVersionUID  int  i  char[]  chars  
[P5_Replace_Variable]^for  ( charsnt i = i.length - 1; i >= 0; i-- )  {^126^^^^^123^132^for  ( int i = chars.length - 1; i >= 0; i-- )  {^[CLASS] CaseInsensitiveMap  [METHOD] convertKey [RETURN_TYPE] Object   Object key [VARIABLES] Object  key  boolean  long  serialVersionUID  int  i  char[]  chars  
[P2_Replace_Operator]^for  ( int i = chars.length  ==  1; i >= 0; i-- )  {^126^^^^^123^132^for  ( int i = chars.length - 1; i >= 0; i-- )  {^[CLASS] CaseInsensitiveMap  [METHOD] convertKey [RETURN_TYPE] Object   Object key [VARIABLES] Object  key  boolean  long  serialVersionUID  int  i  char[]  chars  
[P7_Replace_Invocation]^chars[i] = Character.toLowerCase ( Character .toLowerCase ( null )   ) ;^127^^^^^123^132^chars[i] = Character.toLowerCase ( Character.toUpperCase ( chars[i] )  ) ;^[CLASS] CaseInsensitiveMap  [METHOD] convertKey [RETURN_TYPE] Object   Object key [VARIABLES] Object  key  boolean  long  serialVersionUID  int  i  char[]  chars  
[P7_Replace_Invocation]^return  ( CaseInsensitiveMap<K, V> )  super .toString (  )  ;^142^^^^^141^143^return  ( CaseInsensitiveMap<K, V> )  super.clone (  ) ;^[CLASS] CaseInsensitiveMap  [METHOD] clone [RETURN_TYPE] CaseInsensitiveMap   [VARIABLES] long  serialVersionUID  boolean  
[P14_Delete_Statement]^^142^^^^^141^143^return  ( CaseInsensitiveMap<K, V> )  super.clone (  ) ;^[CLASS] CaseInsensitiveMap  [METHOD] clone [RETURN_TYPE] CaseInsensitiveMap   [VARIABLES] long  serialVersionUID  boolean  
[P14_Delete_Statement]^^149^^^^^148^151^out.defaultWriteObject (  ) ;^[CLASS] CaseInsensitiveMap  [METHOD] writeObject [RETURN_TYPE] void   ObjectOutputStream out [VARIABLES] ObjectOutputStream  out  long  serialVersionUID  boolean  
[P11_Insert_Donor_Statement]^in.defaultReadObject (  ) ;out.defaultWriteObject (  ) ;^149^^^^^148^151^out.defaultWriteObject (  ) ;^[CLASS] CaseInsensitiveMap  [METHOD] writeObject [RETURN_TYPE] void   ObjectOutputStream out [VARIABLES] ObjectOutputStream  out  long  serialVersionUID  boolean  
[P11_Insert_Donor_Statement]^doWriteObject ( out ) ;out.defaultWriteObject (  ) ;^149^^^^^148^151^out.defaultWriteObject (  ) ;^[CLASS] CaseInsensitiveMap  [METHOD] writeObject [RETURN_TYPE] void   ObjectOutputStream out [VARIABLES] ObjectOutputStream  out  long  serialVersionUID  boolean  
[P7_Replace_Invocation]^writeObject ( out ) ;^150^^^^^148^151^doWriteObject ( out ) ;^[CLASS] CaseInsensitiveMap  [METHOD] writeObject [RETURN_TYPE] void   ObjectOutputStream out [VARIABLES] ObjectOutputStream  out  long  serialVersionUID  boolean  
[P14_Delete_Statement]^^150^^^^^148^151^doWriteObject ( out ) ;^[CLASS] CaseInsensitiveMap  [METHOD] writeObject [RETURN_TYPE] void   ObjectOutputStream out [VARIABLES] ObjectOutputStream  out  long  serialVersionUID  boolean  
[P11_Insert_Donor_Statement]^out.defaultWriteObject (  ) ;doWriteObject ( out ) ;^150^^^^^148^151^doWriteObject ( out ) ;^[CLASS] CaseInsensitiveMap  [METHOD] writeObject [RETURN_TYPE] void   ObjectOutputStream out [VARIABLES] ObjectOutputStream  out  long  serialVersionUID  boolean  
[P11_Insert_Donor_Statement]^doReadObject ( in ) ;doWriteObject ( out ) ;^150^^^^^148^151^doWriteObject ( out ) ;^[CLASS] CaseInsensitiveMap  [METHOD] writeObject [RETURN_TYPE] void   ObjectOutputStream out [VARIABLES] ObjectOutputStream  out  long  serialVersionUID  boolean  
[P14_Delete_Statement]^^157^^^^^156^159^in.defaultReadObject (  ) ;^[CLASS] CaseInsensitiveMap  [METHOD] readObject [RETURN_TYPE] void   ObjectInputStream in [VARIABLES] long  serialVersionUID  ObjectInputStream  in  boolean  
[P11_Insert_Donor_Statement]^out.defaultWriteObject (  ) ;in.defaultReadObject (  ) ;^157^^^^^156^159^in.defaultReadObject (  ) ;^[CLASS] CaseInsensitiveMap  [METHOD] readObject [RETURN_TYPE] void   ObjectInputStream in [VARIABLES] long  serialVersionUID  ObjectInputStream  in  boolean  
[P11_Insert_Donor_Statement]^doReadObject ( in ) ;in.defaultReadObject (  ) ;^157^^^^^156^159^in.defaultReadObject (  ) ;^[CLASS] CaseInsensitiveMap  [METHOD] readObject [RETURN_TYPE] void   ObjectInputStream in [VARIABLES] long  serialVersionUID  ObjectInputStream  in  boolean  
[P7_Replace_Invocation]^readObject ( in ) ;^158^^^^^156^159^doReadObject ( in ) ;^[CLASS] CaseInsensitiveMap  [METHOD] readObject [RETURN_TYPE] void   ObjectInputStream in [VARIABLES] long  serialVersionUID  ObjectInputStream  in  boolean  
[P14_Delete_Statement]^^158^^^^^156^159^doReadObject ( in ) ;^[CLASS] CaseInsensitiveMap  [METHOD] readObject [RETURN_TYPE] void   ObjectInputStream in [VARIABLES] long  serialVersionUID  ObjectInputStream  in  boolean  
[P11_Insert_Donor_Statement]^in.defaultReadObject (  ) ;doReadObject ( in ) ;^158^^^^^156^159^doReadObject ( in ) ;^[CLASS] CaseInsensitiveMap  [METHOD] readObject [RETURN_TYPE] void   ObjectInputStream in [VARIABLES] long  serialVersionUID  ObjectInputStream  in  boolean  
[P11_Insert_Donor_Statement]^doWriteObject ( out ) ;doReadObject ( in ) ;^158^^^^^156^159^doReadObject ( in ) ;^[CLASS] CaseInsensitiveMap  [METHOD] readObject [RETURN_TYPE] void   ObjectInputStream in [VARIABLES] long  serialVersionUID  ObjectInputStream  in  boolean  
