[BugLab_Wrong_Operator]^if  ( data != null )  {^100^^^^^99^107^if  ( data == null )  {^[CLASS] DefaultPieDataset  [METHOD] <init> [RETURN_TYPE] KeyedValues)   KeyedValues data [VARIABLES] KeyedValues  data  boolean  DefaultKeyedValues  data  long  serialVersionUID  int  i  
[BugLab_Argument_Swapping]^for  ( datant i = 0; i < i.getItemCount (  ) ; i++ )  {^104^^^^^99^107^for  ( int i = 0; i < data.getItemCount (  ) ; i++ )  {^[CLASS] DefaultPieDataset  [METHOD] <init> [RETURN_TYPE] KeyedValues)   KeyedValues data [VARIABLES] KeyedValues  data  boolean  DefaultKeyedValues  data  long  serialVersionUID  int  i  
[BugLab_Wrong_Operator]^for  ( int i = 0; i == data.getItemCount (  ) ; i++ )  {^104^^^^^99^107^for  ( int i = 0; i < data.getItemCount (  ) ; i++ )  {^[CLASS] DefaultPieDataset  [METHOD] <init> [RETURN_TYPE] KeyedValues)   KeyedValues data [VARIABLES] KeyedValues  data  boolean  DefaultKeyedValues  data  long  serialVersionUID  int  i  
[BugLab_Wrong_Literal]^for  ( int i = i; i < data.getItemCount (  ) ; i++ )  {^104^^^^^99^107^for  ( int i = 0; i < data.getItemCount (  ) ; i++ )  {^[CLASS] DefaultPieDataset  [METHOD] <init> [RETURN_TYPE] KeyedValues)   KeyedValues data [VARIABLES] KeyedValues  data  boolean  DefaultKeyedValues  data  long  serialVersionUID  int  i  
[BugLab_Argument_Swapping]^this.data.addValue ( i.getKey ( data ) , data.getValue ( i )  ) ;^105^^^^^99^107^this.data.addValue ( data.getKey ( i ) , data.getValue ( i )  ) ;^[CLASS] DefaultPieDataset  [METHOD] <init> [RETURN_TYPE] KeyedValues)   KeyedValues data [VARIABLES] KeyedValues  data  boolean  DefaultKeyedValues  data  long  serialVersionUID  int  i  
[BugLab_Wrong_Literal]^for  ( int i = 1; i < data.getItemCount (  ) ; i++ )  {^104^^^^^99^107^for  ( int i = 0; i < data.getItemCount (  ) ; i++ )  {^[CLASS] DefaultPieDataset  [METHOD] <init> [RETURN_TYPE] KeyedValues)   KeyedValues data [VARIABLES] KeyedValues  data  boolean  DefaultKeyedValues  data  long  serialVersionUID  int  i  
[BugLab_Variable_Misuse]^return data.getItemCount (  ) ;^115^^^^^114^116^return this.data.getItemCount (  ) ;^[CLASS] DefaultPieDataset  [METHOD] getItemCount [RETURN_TYPE] int   [VARIABLES] DefaultKeyedValues  data  long  serialVersionUID  boolean  
[BugLab_Variable_Misuse]^return Collections.unmodifiableList ( data.getKeys (  )  ) ;^125^^^^^124^126^return Collections.unmodifiableList ( this.data.getKeys (  )  ) ;^[CLASS] DefaultPieDataset  [METHOD] getKeys [RETURN_TYPE] List   [VARIABLES] DefaultKeyedValues  data  long  serialVersionUID  boolean  
[BugLab_Variable_Misuse]^return data.getKey ( item ) ;^140^^^^^139^141^return this.data.getKey ( item ) ;^[CLASS] DefaultPieDataset  [METHOD] getKey [RETURN_TYPE] Comparable   int item [VARIABLES] boolean  DefaultKeyedValues  data  long  serialVersionUID  int  item  
[BugLab_Argument_Swapping]^return item.getKey ( this.data ) ;^140^^^^^139^141^return this.data.getKey ( item ) ;^[CLASS] DefaultPieDataset  [METHOD] getKey [RETURN_TYPE] Comparable   int item [VARIABLES] boolean  DefaultKeyedValues  data  long  serialVersionUID  int  item  
[BugLab_Variable_Misuse]^return data.getIndex ( key ) ;^154^^^^^153^155^return this.data.getIndex ( key ) ;^[CLASS] DefaultPieDataset  [METHOD] getIndex [RETURN_TYPE] int   Comparable key [VARIABLES] Comparable  key  boolean  DefaultKeyedValues  data  long  serialVersionUID  
[BugLab_Argument_Swapping]^return key.getIndex ( this.data ) ;^154^^^^^153^155^return this.data.getIndex ( key ) ;^[CLASS] DefaultPieDataset  [METHOD] getIndex [RETURN_TYPE] int   Comparable key [VARIABLES] Comparable  key  boolean  DefaultKeyedValues  data  long  serialVersionUID  
[BugLab_Wrong_Operator]^if  ( getItemCount (  )  >= item )  {^167^^^^^164^172^if  ( getItemCount (  )  > item )  {^[CLASS] DefaultPieDataset  [METHOD] getValue [RETURN_TYPE] Number   int item [VARIABLES] boolean  DefaultKeyedValues  data  Number  result  long  serialVersionUID  int  item  
[BugLab_Variable_Misuse]^result = data.getValue ( item ) ;^168^^^^^164^172^result = this.data.getValue ( item ) ;^[CLASS] DefaultPieDataset  [METHOD] getValue [RETURN_TYPE] Number   int item [VARIABLES] boolean  DefaultKeyedValues  data  Number  result  long  serialVersionUID  int  item  
[BugLab_Argument_Swapping]^result = item.getValue ( this.data ) ;^168^^^^^164^172^result = this.data.getValue ( item ) ;^[CLASS] DefaultPieDataset  [METHOD] getValue [RETURN_TYPE] Number   int item [VARIABLES] boolean  DefaultKeyedValues  data  Number  result  long  serialVersionUID  int  item  
[BugLab_Wrong_Operator]^if  ( key != null )  {^184^^^^^183^188^if  ( key == null )  {^[CLASS] DefaultPieDataset  [METHOD] getValue [RETURN_TYPE] Number   Comparable key [VARIABLES] Comparable  key  boolean  DefaultKeyedValues  data  long  serialVersionUID  
[BugLab_Argument_Swapping]^return key.getValue ( this.data ) ;^187^^^^^183^188^return this.data.getValue ( key ) ;^[CLASS] DefaultPieDataset  [METHOD] getValue [RETURN_TYPE] Number   Comparable key [VARIABLES] Comparable  key  boolean  DefaultKeyedValues  data  long  serialVersionUID  
[BugLab_Variable_Misuse]^return data.getValue ( key ) ;^187^^^^^183^188^return this.data.getValue ( key ) ;^[CLASS] DefaultPieDataset  [METHOD] getValue [RETURN_TYPE] Number   Comparable key [VARIABLES] Comparable  key  boolean  DefaultKeyedValues  data  long  serialVersionUID  
[BugLab_Argument_Swapping]^this.data.setValue ( value, key ) ;^201^^^^^200^203^this.data.setValue ( key, value ) ;^[CLASS] DefaultPieDataset  [METHOD] setValue [RETURN_TYPE] void   Comparable key Number value [VARIABLES] Comparable  key  boolean  DefaultKeyedValues  data  Number  value  long  serialVersionUID  
[BugLab_Argument_Swapping]^setValue ( value, new Double ( key )  ) ;^216^^^^^215^217^setValue ( key, new Double ( value )  ) ;^[CLASS] DefaultPieDataset  [METHOD] setValue [RETURN_TYPE] void   Comparable key double value [VARIABLES] Comparable  key  boolean  DefaultKeyedValues  data  long  serialVersionUID  double  value  
[BugLab_Argument_Swapping]^insertValue ( value, key, new Double ( position )  ) ;^233^^^^^232^234^insertValue ( position, key, new Double ( value )  ) ;^[CLASS] DefaultPieDataset  [METHOD] insertValue [RETURN_TYPE] void   int position Comparable key double value [VARIABLES] Comparable  key  boolean  DefaultKeyedValues  data  long  serialVersionUID  int  position  double  value  
[BugLab_Argument_Swapping]^insertValue ( position, value, new Double ( key )  ) ;^233^^^^^232^234^insertValue ( position, key, new Double ( value )  ) ;^[CLASS] DefaultPieDataset  [METHOD] insertValue [RETURN_TYPE] void   int position Comparable key double value [VARIABLES] Comparable  key  boolean  DefaultKeyedValues  data  long  serialVersionUID  int  position  double  value  
[BugLab_Argument_Swapping]^this.data.insertValue ( key, position, value ) ;^250^^^^^249^252^this.data.insertValue ( position, key, value ) ;^[CLASS] DefaultPieDataset  [METHOD] insertValue [RETURN_TYPE] void   int position Comparable key Number value [VARIABLES] Comparable  key  boolean  DefaultKeyedValues  data  Number  value  long  serialVersionUID  int  position  
[BugLab_Argument_Swapping]^this.data.insertValue ( value, key, position ) ;^250^^^^^249^252^this.data.insertValue ( position, key, value ) ;^[CLASS] DefaultPieDataset  [METHOD] insertValue [RETURN_TYPE] void   int position Comparable key Number value [VARIABLES] Comparable  key  boolean  DefaultKeyedValues  data  Number  value  long  serialVersionUID  int  position  
[BugLab_Wrong_Operator]^if  ( getItemCount (  )  >= 0 )  {^275^^^^^274^279^if  ( getItemCount (  )  > 0 )  {^[CLASS] DefaultPieDataset  [METHOD] clear [RETURN_TYPE] void   [VARIABLES] DefaultKeyedValues  data  long  serialVersionUID  boolean  
[BugLab_Wrong_Operator]^if  ( obj < this )  {^315^^^^^300^330^if  ( obj == this )  {^[CLASS] DefaultPieDataset  [METHOD] equals [RETURN_TYPE] boolean   Object obj [VARIABLES] Comparable  k1  k2  boolean  DefaultKeyedValues  data  Number  v1  v2  PieDataset  that  Object  obj  long  serialVersionUID  int  count  i  
[BugLab_Wrong_Literal]^return false;^316^^^^^301^331^return true;^[CLASS] DefaultPieDataset  [METHOD] equals [RETURN_TYPE] boolean   Object obj [VARIABLES] Comparable  k1  k2  boolean  DefaultKeyedValues  data  Number  v1  v2  PieDataset  that  Object  obj  long  serialVersionUID  int  count  i  
[BugLab_Wrong_Operator]^if  ( ! ( obj  ^  PieDataset )  )  {^319^^^^^304^334^if  ( ! ( obj instanceof PieDataset )  )  {^[CLASS] DefaultPieDataset  [METHOD] equals [RETURN_TYPE] boolean   Object obj [VARIABLES] Comparable  k1  k2  boolean  DefaultKeyedValues  data  Number  v1  v2  PieDataset  that  Object  obj  long  serialVersionUID  int  count  i  
[BugLab_Wrong_Literal]^return true;^320^^^^^305^335^return false;^[CLASS] DefaultPieDataset  [METHOD] equals [RETURN_TYPE] boolean   Object obj [VARIABLES] Comparable  k1  k2  boolean  DefaultKeyedValues  data  Number  v1  v2  PieDataset  that  Object  obj  long  serialVersionUID  int  count  i  
[BugLab_Variable_Misuse]^if  ( that.getItemCount (  )  != i )  {^324^^^^^309^339^if  ( that.getItemCount (  )  != count )  {^[CLASS] DefaultPieDataset  [METHOD] equals [RETURN_TYPE] boolean   Object obj [VARIABLES] Comparable  k1  k2  boolean  DefaultKeyedValues  data  Number  v1  v2  PieDataset  that  Object  obj  long  serialVersionUID  int  count  i  
[BugLab_Argument_Swapping]^if  ( count.getItemCount (  )  != that )  {^324^^^^^309^339^if  ( that.getItemCount (  )  != count )  {^[CLASS] DefaultPieDataset  [METHOD] equals [RETURN_TYPE] boolean   Object obj [VARIABLES] Comparable  k1  k2  boolean  DefaultKeyedValues  data  Number  v1  v2  PieDataset  that  Object  obj  long  serialVersionUID  int  count  i  
[BugLab_Wrong_Operator]^if  ( that.getItemCount (  )  == count )  {^324^^^^^309^339^if  ( that.getItemCount (  )  != count )  {^[CLASS] DefaultPieDataset  [METHOD] equals [RETURN_TYPE] boolean   Object obj [VARIABLES] Comparable  k1  k2  boolean  DefaultKeyedValues  data  Number  v1  v2  PieDataset  that  Object  obj  long  serialVersionUID  int  count  i  
[BugLab_Wrong_Literal]^return true;^325^^^^^310^340^return false;^[CLASS] DefaultPieDataset  [METHOD] equals [RETURN_TYPE] boolean   Object obj [VARIABLES] Comparable  k1  k2  boolean  DefaultKeyedValues  data  Number  v1  v2  PieDataset  that  Object  obj  long  serialVersionUID  int  count  i  
[BugLab_Wrong_Literal]^return true;^332^^^^^317^347^return false;^[CLASS] DefaultPieDataset  [METHOD] equals [RETURN_TYPE] boolean   Object obj [VARIABLES] Comparable  k1  k2  boolean  DefaultKeyedValues  data  Number  v1  v2  PieDataset  that  Object  obj  long  serialVersionUID  int  count  i  
[BugLab_Wrong_Operator]^if  ( v1 != null )  {^337^^^^^322^352^if  ( v1 == null )  {^[CLASS] DefaultPieDataset  [METHOD] equals [RETURN_TYPE] boolean   Object obj [VARIABLES] Comparable  k1  k2  boolean  DefaultKeyedValues  data  Number  v1  v2  PieDataset  that  Object  obj  long  serialVersionUID  int  count  i  
[BugLab_Wrong_Literal]^return true;^344^^^^^337^346^return false;^[CLASS] DefaultPieDataset  [METHOD] equals [RETURN_TYPE] boolean   Object obj [VARIABLES] Comparable  k1  k2  boolean  DefaultKeyedValues  data  Number  v1  v2  PieDataset  that  Object  obj  long  serialVersionUID  int  count  i  
[BugLab_Variable_Misuse]^if  ( v1 != null )  {^338^^^^^323^353^if  ( v2 != null )  {^[CLASS] DefaultPieDataset  [METHOD] equals [RETURN_TYPE] boolean   Object obj [VARIABLES] Comparable  k1  k2  boolean  DefaultKeyedValues  data  Number  v1  v2  PieDataset  that  Object  obj  long  serialVersionUID  int  count  i  
[BugLab_Wrong_Operator]^if  ( v2 == null )  {^338^^^^^323^353^if  ( v2 != null )  {^[CLASS] DefaultPieDataset  [METHOD] equals [RETURN_TYPE] boolean   Object obj [VARIABLES] Comparable  k1  k2  boolean  DefaultKeyedValues  data  Number  v1  v2  PieDataset  that  Object  obj  long  serialVersionUID  int  count  i  
[BugLab_Wrong_Literal]^return true;^339^^^^^324^354^return false;^[CLASS] DefaultPieDataset  [METHOD] equals [RETURN_TYPE] boolean   Object obj [VARIABLES] Comparable  k1  k2  boolean  DefaultKeyedValues  data  Number  v1  v2  PieDataset  that  Object  obj  long  serialVersionUID  int  count  i  
[BugLab_Wrong_Literal]^return true;^344^^^^^329^359^return false;^[CLASS] DefaultPieDataset  [METHOD] equals [RETURN_TYPE] boolean   Object obj [VARIABLES] Comparable  k1  k2  boolean  DefaultKeyedValues  data  Number  v1  v2  PieDataset  that  Object  obj  long  serialVersionUID  int  count  i  
[BugLab_Variable_Misuse]^for  ( countnt i = 0; i < count; i++ )  {^328^^^^^313^343^for  ( int i = 0; i < count; i++ )  {^[CLASS] DefaultPieDataset  [METHOD] equals [RETURN_TYPE] boolean   Object obj [VARIABLES] Comparable  k1  k2  boolean  DefaultKeyedValues  data  Number  v1  v2  PieDataset  that  Object  obj  long  serialVersionUID  int  count  i  
[BugLab_Argument_Swapping]^for  ( countnt i = 0; i < i; i++ )  {^328^^^^^313^343^for  ( int i = 0; i < count; i++ )  {^[CLASS] DefaultPieDataset  [METHOD] equals [RETURN_TYPE] boolean   Object obj [VARIABLES] Comparable  k1  k2  boolean  DefaultKeyedValues  data  Number  v1  v2  PieDataset  that  Object  obj  long  serialVersionUID  int  count  i  
[BugLab_Wrong_Operator]^for  ( int i = 0; i <= count; i++ )  {^328^^^^^313^343^for  ( int i = 0; i < count; i++ )  {^[CLASS] DefaultPieDataset  [METHOD] equals [RETURN_TYPE] boolean   Object obj [VARIABLES] Comparable  k1  k2  boolean  DefaultKeyedValues  data  Number  v1  v2  PieDataset  that  Object  obj  long  serialVersionUID  int  count  i  
[BugLab_Wrong_Literal]^for  ( int i = i; i < count; i++ )  {^328^^^^^313^343^for  ( int i = 0; i < count; i++ )  {^[CLASS] DefaultPieDataset  [METHOD] equals [RETURN_TYPE] boolean   Object obj [VARIABLES] Comparable  k1  k2  boolean  DefaultKeyedValues  data  Number  v1  v2  PieDataset  that  Object  obj  long  serialVersionUID  int  count  i  
[BugLab_Variable_Misuse]^Comparable k1 = getKey ( count ) ;^329^^^^^314^344^Comparable k1 = getKey ( i ) ;^[CLASS] DefaultPieDataset  [METHOD] equals [RETURN_TYPE] boolean   Object obj [VARIABLES] Comparable  k1  k2  boolean  DefaultKeyedValues  data  Number  v1  v2  PieDataset  that  Object  obj  long  serialVersionUID  int  count  i  
[BugLab_Variable_Misuse]^Comparable k2 = that.getKey ( count ) ;^330^^^^^315^345^Comparable k2 = that.getKey ( i ) ;^[CLASS] DefaultPieDataset  [METHOD] equals [RETURN_TYPE] boolean   Object obj [VARIABLES] Comparable  k1  k2  boolean  DefaultKeyedValues  data  Number  v1  v2  PieDataset  that  Object  obj  long  serialVersionUID  int  count  i  
[BugLab_Argument_Swapping]^Comparable k2 = i.getKey ( that ) ;^330^^^^^315^345^Comparable k2 = that.getKey ( i ) ;^[CLASS] DefaultPieDataset  [METHOD] equals [RETURN_TYPE] boolean   Object obj [VARIABLES] Comparable  k1  k2  boolean  DefaultKeyedValues  data  Number  v1  v2  PieDataset  that  Object  obj  long  serialVersionUID  int  count  i  
[BugLab_Variable_Misuse]^Number v1 = getValue ( count ) ;^335^^^^^320^350^Number v1 = getValue ( i ) ;^[CLASS] DefaultPieDataset  [METHOD] equals [RETURN_TYPE] boolean   Object obj [VARIABLES] Comparable  k1  k2  boolean  DefaultKeyedValues  data  Number  v1  v2  PieDataset  that  Object  obj  long  serialVersionUID  int  count  i  
[BugLab_Variable_Misuse]^Number v2 = that.getValue ( count ) ;^336^^^^^321^351^Number v2 = that.getValue ( i ) ;^[CLASS] DefaultPieDataset  [METHOD] equals [RETURN_TYPE] boolean   Object obj [VARIABLES] Comparable  k1  k2  boolean  DefaultKeyedValues  data  Number  v1  v2  PieDataset  that  Object  obj  long  serialVersionUID  int  count  i  
[BugLab_Argument_Swapping]^Number v2 = i.getValue ( that ) ;^336^^^^^321^351^Number v2 = that.getValue ( i ) ;^[CLASS] DefaultPieDataset  [METHOD] equals [RETURN_TYPE] boolean   Object obj [VARIABLES] Comparable  k1  k2  boolean  DefaultKeyedValues  data  Number  v1  v2  PieDataset  that  Object  obj  long  serialVersionUID  int  count  i  
[BugLab_Wrong_Literal]^for  ( int i = count; i < count; i++ )  {^328^^^^^313^343^for  ( int i = 0; i < count; i++ )  {^[CLASS] DefaultPieDataset  [METHOD] equals [RETURN_TYPE] boolean   Object obj [VARIABLES] Comparable  k1  k2  boolean  DefaultKeyedValues  data  Number  v1  v2  PieDataset  that  Object  obj  long  serialVersionUID  int  count  i  
[BugLab_Wrong_Literal]^return false;^348^^^^^333^363^return true;^[CLASS] DefaultPieDataset  [METHOD] equals [RETURN_TYPE] boolean   Object obj [VARIABLES] Comparable  k1  k2  boolean  DefaultKeyedValues  data  Number  v1  v2  PieDataset  that  Object  obj  long  serialVersionUID  int  count  i  
[BugLab_Variable_Misuse]^return data.hashCode (  ) ;^358^^^^^357^359^return this.data.hashCode (  ) ;^[CLASS] DefaultPieDataset  [METHOD] hashCode [RETURN_TYPE] int   [VARIABLES] DefaultKeyedValues  data  long  serialVersionUID  boolean  
[BugLab_Variable_Misuse]^clone.data =  ( DefaultKeyedValues )  data.clone (  ) ;^371^^^^^369^373^clone.data =  ( DefaultKeyedValues )  this.data.clone (  ) ;^[CLASS] DefaultPieDataset  [METHOD] clone [RETURN_TYPE] Object   [VARIABLES] DefaultPieDataset  clone  boolean  DefaultKeyedValues  data  long  serialVersionUID  