[BugLab_Wrong_Operator]^if  ( labelFormat != null )  {^86^^^^^82^101^if  ( labelFormat == null )  {^[CLASS] AbstractPieItemLabelGenerator  [METHOD] <init> [RETURN_TYPE] NumberFormat)   String labelFormat NumberFormat numberFormat NumberFormat percentFormat [VARIABLES] String  labelFormat  boolean  long  serialVersionUID  NumberFormat  numberFormat  percentFormat  
[BugLab_Variable_Misuse]^if  ( percentFormat == null )  {^89^^^^^82^101^if  ( numberFormat == null )  {^[CLASS] AbstractPieItemLabelGenerator  [METHOD] <init> [RETURN_TYPE] NumberFormat)   String labelFormat NumberFormat numberFormat NumberFormat percentFormat [VARIABLES] String  labelFormat  boolean  long  serialVersionUID  NumberFormat  numberFormat  percentFormat  
[BugLab_Wrong_Operator]^if  ( numberFormat != null )  {^89^^^^^82^101^if  ( numberFormat == null )  {^[CLASS] AbstractPieItemLabelGenerator  [METHOD] <init> [RETURN_TYPE] NumberFormat)   String labelFormat NumberFormat numberFormat NumberFormat percentFormat [VARIABLES] String  labelFormat  boolean  long  serialVersionUID  NumberFormat  numberFormat  percentFormat  
[BugLab_Variable_Misuse]^if  ( numberFormat == null )  {^92^^^^^82^101^if  ( percentFormat == null )  {^[CLASS] AbstractPieItemLabelGenerator  [METHOD] <init> [RETURN_TYPE] NumberFormat)   String labelFormat NumberFormat numberFormat NumberFormat percentFormat [VARIABLES] String  labelFormat  boolean  long  serialVersionUID  NumberFormat  numberFormat  percentFormat  
[BugLab_Wrong_Operator]^if  ( percentFormat != null )  {^92^^^^^82^101^if  ( percentFormat == null )  {^[CLASS] AbstractPieItemLabelGenerator  [METHOD] <init> [RETURN_TYPE] NumberFormat)   String labelFormat NumberFormat numberFormat NumberFormat percentFormat [VARIABLES] String  labelFormat  boolean  long  serialVersionUID  NumberFormat  numberFormat  percentFormat  
[BugLab_Variable_Misuse]^this.numberFormat = percentFormat;^98^^^^^82^101^this.numberFormat = numberFormat;^[CLASS] AbstractPieItemLabelGenerator  [METHOD] <init> [RETURN_TYPE] NumberFormat)   String labelFormat NumberFormat numberFormat NumberFormat percentFormat [VARIABLES] String  labelFormat  boolean  long  serialVersionUID  NumberFormat  numberFormat  percentFormat  
[BugLab_Variable_Misuse]^this.percentFormat = numberFormat;^99^^^^^82^101^this.percentFormat = percentFormat;^[CLASS] AbstractPieItemLabelGenerator  [METHOD] <init> [RETURN_TYPE] NumberFormat)   String labelFormat NumberFormat numberFormat NumberFormat percentFormat [VARIABLES] String  labelFormat  boolean  long  serialVersionUID  NumberFormat  numberFormat  percentFormat  
[BugLab_Variable_Misuse]^return labelFormat;^109^^^^^108^110^return this.labelFormat;^[CLASS] AbstractPieItemLabelGenerator  [METHOD] getLabelFormat [RETURN_TYPE] String   [VARIABLES] String  labelFormat  boolean  long  serialVersionUID  NumberFormat  numberFormat  percentFormat  
[BugLab_Variable_Misuse]^return percentFormat;^118^^^^^117^119^return this.numberFormat;^[CLASS] AbstractPieItemLabelGenerator  [METHOD] getNumberFormat [RETURN_TYPE] NumberFormat   [VARIABLES] String  labelFormat  boolean  long  serialVersionUID  NumberFormat  numberFormat  percentFormat  
[BugLab_Variable_Misuse]^return percentFormat;^127^^^^^126^128^return this.percentFormat;^[CLASS] AbstractPieItemLabelGenerator  [METHOD] getPercentFormat [RETURN_TYPE] NumberFormat   [VARIABLES] String  labelFormat  boolean  long  serialVersionUID  NumberFormat  numberFormat  percentFormat  
[BugLab_Wrong_Literal]^Object[] result = new Object[5];^147^^^^^146^167^Object[] result = new Object[4];^[CLASS] AbstractPieItemLabelGenerator  [METHOD] createItemArray [RETURN_TYPE] Object[]   PieDataset dataset Comparable key [VARIABLES] Comparable  key  boolean  Number  value  PieDataset  dataset  double  percent  total  v  String  labelFormat  long  serialVersionUID  Object[]  result  NumberFormat  numberFormat  percentFormat  
[BugLab_Wrong_Literal]^result[3] = key.toString (  ) ;^149^^^^^146^167^result[0] = key.toString (  ) ;^[CLASS] AbstractPieItemLabelGenerator  [METHOD] createItemArray [RETURN_TYPE] Object[]   PieDataset dataset Comparable key [VARIABLES] Comparable  key  boolean  Number  value  PieDataset  dataset  double  percent  total  v  String  labelFormat  long  serialVersionUID  Object[]  result  NumberFormat  numberFormat  percentFormat  
[BugLab_Argument_Swapping]^Number value = key.getValue ( dataset ) ;^150^^^^^146^167^Number value = dataset.getValue ( key ) ;^[CLASS] AbstractPieItemLabelGenerator  [METHOD] createItemArray [RETURN_TYPE] Object[]   PieDataset dataset Comparable key [VARIABLES] Comparable  key  boolean  Number  value  PieDataset  dataset  double  percent  total  v  String  labelFormat  long  serialVersionUID  Object[]  result  NumberFormat  numberFormat  percentFormat  
[BugLab_Wrong_Operator]^if  ( value == null )  {^151^^^^^146^167^if  ( value != null )  {^[CLASS] AbstractPieItemLabelGenerator  [METHOD] createItemArray [RETURN_TYPE] Object[]   PieDataset dataset Comparable key [VARIABLES] Comparable  key  boolean  Number  value  PieDataset  dataset  double  percent  total  v  String  labelFormat  long  serialVersionUID  Object[]  result  NumberFormat  numberFormat  percentFormat  
[BugLab_Variable_Misuse]^result[1] = percentFormat.format ( value ) ;^152^^^^^146^167^result[1] = this.numberFormat.format ( value ) ;^[CLASS] AbstractPieItemLabelGenerator  [METHOD] createItemArray [RETURN_TYPE] Object[]   PieDataset dataset Comparable key [VARIABLES] Comparable  key  boolean  Number  value  PieDataset  dataset  double  percent  total  v  String  labelFormat  long  serialVersionUID  Object[]  result  NumberFormat  numberFormat  percentFormat  
[BugLab_Argument_Swapping]^result[1] = value.format ( this.numberFormat ) ;^152^^^^^146^167^result[1] = this.numberFormat.format ( value ) ;^[CLASS] AbstractPieItemLabelGenerator  [METHOD] createItemArray [RETURN_TYPE] Object[]   PieDataset dataset Comparable key [VARIABLES] Comparable  key  boolean  Number  value  PieDataset  dataset  double  percent  total  v  String  labelFormat  long  serialVersionUID  Object[]  result  NumberFormat  numberFormat  percentFormat  
[BugLab_Wrong_Literal]^result[this] = this.numberFormat.format ( value ) ;^152^^^^^146^167^result[1] = this.numberFormat.format ( value ) ;^[CLASS] AbstractPieItemLabelGenerator  [METHOD] createItemArray [RETURN_TYPE] Object[]   PieDataset dataset Comparable key [VARIABLES] Comparable  key  boolean  Number  value  PieDataset  dataset  double  percent  total  v  String  labelFormat  long  serialVersionUID  Object[]  result  NumberFormat  numberFormat  percentFormat  
[BugLab_Wrong_Literal]^result[2] = "null";^155^^^^^146^167^result[1] = "null";^[CLASS] AbstractPieItemLabelGenerator  [METHOD] createItemArray [RETURN_TYPE] Object[]   PieDataset dataset Comparable key [VARIABLES] Comparable  key  boolean  Number  value  PieDataset  dataset  double  percent  total  v  String  labelFormat  long  serialVersionUID  Object[]  result  NumberFormat  numberFormat  percentFormat  
[BugLab_Variable_Misuse]^if  ( totalalue != null )  {^158^^^^^146^167^if  ( value != null )  {^[CLASS] AbstractPieItemLabelGenerator  [METHOD] createItemArray [RETURN_TYPE] Object[]   PieDataset dataset Comparable key [VARIABLES] Comparable  key  boolean  Number  value  PieDataset  dataset  double  percent  total  v  String  labelFormat  long  serialVersionUID  Object[]  result  NumberFormat  numberFormat  percentFormat  
[BugLab_Wrong_Operator]^if  ( value == null )  {^158^^^^^146^167^if  ( value != null )  {^[CLASS] AbstractPieItemLabelGenerator  [METHOD] createItemArray [RETURN_TYPE] Object[]   PieDataset dataset Comparable key [VARIABLES] Comparable  key  boolean  Number  value  PieDataset  dataset  double  percent  total  v  String  labelFormat  long  serialVersionUID  Object[]  result  NumberFormat  numberFormat  percentFormat  
[BugLab_Wrong_Operator]^if  ( v >= 0.0 )  {^160^^^^^146^167^if  ( v > 0.0 )  {^[CLASS] AbstractPieItemLabelGenerator  [METHOD] createItemArray [RETURN_TYPE] Object[]   PieDataset dataset Comparable key [VARIABLES] Comparable  key  boolean  Number  value  PieDataset  dataset  double  percent  total  v  String  labelFormat  long  serialVersionUID  Object[]  result  NumberFormat  numberFormat  percentFormat  
[BugLab_Variable_Misuse]^percent = percent / total;^161^^^^^146^167^percent = v / total;^[CLASS] AbstractPieItemLabelGenerator  [METHOD] createItemArray [RETURN_TYPE] Object[]   PieDataset dataset Comparable key [VARIABLES] Comparable  key  boolean  Number  value  PieDataset  dataset  double  percent  total  v  String  labelFormat  long  serialVersionUID  Object[]  result  NumberFormat  numberFormat  percentFormat  
[BugLab_Variable_Misuse]^percent = v / percent;^161^^^^^146^167^percent = v / total;^[CLASS] AbstractPieItemLabelGenerator  [METHOD] createItemArray [RETURN_TYPE] Object[]   PieDataset dataset Comparable key [VARIABLES] Comparable  key  boolean  Number  value  PieDataset  dataset  double  percent  total  v  String  labelFormat  long  serialVersionUID  Object[]  result  NumberFormat  numberFormat  percentFormat  
[BugLab_Argument_Swapping]^percent = total / v;^161^^^^^146^167^percent = v / total;^[CLASS] AbstractPieItemLabelGenerator  [METHOD] createItemArray [RETURN_TYPE] Object[]   PieDataset dataset Comparable key [VARIABLES] Comparable  key  boolean  Number  value  PieDataset  dataset  double  percent  total  v  String  labelFormat  long  serialVersionUID  Object[]  result  NumberFormat  numberFormat  percentFormat  
[BugLab_Wrong_Operator]^percent = v + total;^161^^^^^146^167^percent = v / total;^[CLASS] AbstractPieItemLabelGenerator  [METHOD] createItemArray [RETURN_TYPE] Object[]   PieDataset dataset Comparable key [VARIABLES] Comparable  key  boolean  Number  value  PieDataset  dataset  double  percent  total  v  String  labelFormat  long  serialVersionUID  Object[]  result  NumberFormat  numberFormat  percentFormat  
[BugLab_Variable_Misuse]^if  ( percent > 0.0 )  {^160^^^^^146^167^if  ( v > 0.0 )  {^[CLASS] AbstractPieItemLabelGenerator  [METHOD] createItemArray [RETURN_TYPE] Object[]   PieDataset dataset Comparable key [VARIABLES] Comparable  key  boolean  Number  value  PieDataset  dataset  double  percent  total  v  String  labelFormat  long  serialVersionUID  Object[]  result  NumberFormat  numberFormat  percentFormat  
[BugLab_Wrong_Operator]^percent = v - total;^161^^^^^146^167^percent = v / total;^[CLASS] AbstractPieItemLabelGenerator  [METHOD] createItemArray [RETURN_TYPE] Object[]   PieDataset dataset Comparable key [VARIABLES] Comparable  key  boolean  Number  value  PieDataset  dataset  double  percent  total  v  String  labelFormat  long  serialVersionUID  Object[]  result  NumberFormat  numberFormat  percentFormat  
[BugLab_Variable_Misuse]^result[2] = this.percentFormat.format ( v ) ;^164^^^^^146^167^result[2] = this.percentFormat.format ( percent ) ;^[CLASS] AbstractPieItemLabelGenerator  [METHOD] createItemArray [RETURN_TYPE] Object[]   PieDataset dataset Comparable key [VARIABLES] Comparable  key  boolean  Number  value  PieDataset  dataset  double  percent  total  v  String  labelFormat  long  serialVersionUID  Object[]  result  NumberFormat  numberFormat  percentFormat  
[BugLab_Variable_Misuse]^result[2] = percentFormat.format ( percent ) ;^164^^^^^146^167^result[2] = this.percentFormat.format ( percent ) ;^[CLASS] AbstractPieItemLabelGenerator  [METHOD] createItemArray [RETURN_TYPE] Object[]   PieDataset dataset Comparable key [VARIABLES] Comparable  key  boolean  Number  value  PieDataset  dataset  double  percent  total  v  String  labelFormat  long  serialVersionUID  Object[]  result  NumberFormat  numberFormat  percentFormat  
[BugLab_Argument_Swapping]^result[2] = percent.format ( this.percentFormat ) ;^164^^^^^146^167^result[2] = this.percentFormat.format ( percent ) ;^[CLASS] AbstractPieItemLabelGenerator  [METHOD] createItemArray [RETURN_TYPE] Object[]   PieDataset dataset Comparable key [VARIABLES] Comparable  key  boolean  Number  value  PieDataset  dataset  double  percent  total  v  String  labelFormat  long  serialVersionUID  Object[]  result  NumberFormat  numberFormat  percentFormat  
[BugLab_Wrong_Literal]^result[1] = this.percentFormat.format ( percent ) ;^164^^^^^146^167^result[2] = this.percentFormat.format ( percent ) ;^[CLASS] AbstractPieItemLabelGenerator  [METHOD] createItemArray [RETURN_TYPE] Object[]   PieDataset dataset Comparable key [VARIABLES] Comparable  key  boolean  Number  value  PieDataset  dataset  double  percent  total  v  String  labelFormat  long  serialVersionUID  Object[]  result  NumberFormat  numberFormat  percentFormat  
[BugLab_Variable_Misuse]^result[3] = this.numberFormat.format ( v ) ;^165^^^^^146^167^result[3] = this.numberFormat.format ( total ) ;^[CLASS] AbstractPieItemLabelGenerator  [METHOD] createItemArray [RETURN_TYPE] Object[]   PieDataset dataset Comparable key [VARIABLES] Comparable  key  boolean  Number  value  PieDataset  dataset  double  percent  total  v  String  labelFormat  long  serialVersionUID  Object[]  result  NumberFormat  numberFormat  percentFormat  
[BugLab_Variable_Misuse]^result[3] = percentFormat.format ( total ) ;^165^^^^^146^167^result[3] = this.numberFormat.format ( total ) ;^[CLASS] AbstractPieItemLabelGenerator  [METHOD] createItemArray [RETURN_TYPE] Object[]   PieDataset dataset Comparable key [VARIABLES] Comparable  key  boolean  Number  value  PieDataset  dataset  double  percent  total  v  String  labelFormat  long  serialVersionUID  Object[]  result  NumberFormat  numberFormat  percentFormat  
[BugLab_Argument_Swapping]^result[3] = total.format ( this.numberFormat ) ;^165^^^^^146^167^result[3] = this.numberFormat.format ( total ) ;^[CLASS] AbstractPieItemLabelGenerator  [METHOD] createItemArray [RETURN_TYPE] Object[]   PieDataset dataset Comparable key [VARIABLES] Comparable  key  boolean  Number  value  PieDataset  dataset  double  percent  total  v  String  labelFormat  long  serialVersionUID  Object[]  result  NumberFormat  numberFormat  percentFormat  
[BugLab_Wrong_Literal]^result[2] = this.numberFormat.format ( total ) ;^165^^^^^146^167^result[3] = this.numberFormat.format ( total ) ;^[CLASS] AbstractPieItemLabelGenerator  [METHOD] createItemArray [RETURN_TYPE] Object[]   PieDataset dataset Comparable key [VARIABLES] Comparable  key  boolean  Number  value  PieDataset  dataset  double  percent  total  v  String  labelFormat  long  serialVersionUID  Object[]  result  NumberFormat  numberFormat  percentFormat  
[BugLab_Wrong_Operator]^if  ( dataset == null )  {^179^^^^^177^184^if  ( dataset != null )  {^[CLASS] AbstractPieItemLabelGenerator  [METHOD] generateSectionLabel [RETURN_TYPE] String   PieDataset dataset Comparable key [VARIABLES] Comparable  key  boolean  PieDataset  dataset  String  labelFormat  result  long  serialVersionUID  Object[]  items  NumberFormat  numberFormat  percentFormat  
[BugLab_Variable_Misuse]^result = MessageFormat.format ( result, items ) ;^181^^^^^177^184^result = MessageFormat.format ( this.labelFormat, items ) ;^[CLASS] AbstractPieItemLabelGenerator  [METHOD] generateSectionLabel [RETURN_TYPE] String   PieDataset dataset Comparable key [VARIABLES] Comparable  key  boolean  PieDataset  dataset  String  labelFormat  result  long  serialVersionUID  Object[]  items  NumberFormat  numberFormat  percentFormat  
[BugLab_Argument_Swapping]^result = MessageFormat.format ( items, this.labelFormat ) ;^181^^^^^177^184^result = MessageFormat.format ( this.labelFormat, items ) ;^[CLASS] AbstractPieItemLabelGenerator  [METHOD] generateSectionLabel [RETURN_TYPE] String   PieDataset dataset Comparable key [VARIABLES] Comparable  key  boolean  PieDataset  dataset  String  labelFormat  result  long  serialVersionUID  Object[]  items  NumberFormat  numberFormat  percentFormat  
[BugLab_Argument_Swapping]^Object[] items = createItemArray ( key, dataset ) ;^180^^^^^177^184^Object[] items = createItemArray ( dataset, key ) ;^[CLASS] AbstractPieItemLabelGenerator  [METHOD] generateSectionLabel [RETURN_TYPE] String   PieDataset dataset Comparable key [VARIABLES] Comparable  key  boolean  PieDataset  dataset  String  labelFormat  result  long  serialVersionUID  Object[]  items  NumberFormat  numberFormat  percentFormat  
[BugLab_Variable_Misuse]^return labelFormat;^183^^^^^177^184^return result;^[CLASS] AbstractPieItemLabelGenerator  [METHOD] generateSectionLabel [RETURN_TYPE] String   PieDataset dataset Comparable key [VARIABLES] Comparable  key  boolean  PieDataset  dataset  String  labelFormat  result  long  serialVersionUID  Object[]  items  NumberFormat  numberFormat  percentFormat  
[BugLab_Wrong_Operator]^if  ( obj != this )  {^194^^^^^193^214^if  ( obj == this )  {^[CLASS] AbstractPieItemLabelGenerator  [METHOD] equals [RETURN_TYPE] boolean   Object obj [VARIABLES] AbstractPieItemLabelGenerator  that  Object  obj  String  labelFormat  result  boolean  long  serialVersionUID  NumberFormat  numberFormat  percentFormat  
[BugLab_Wrong_Literal]^return false;^195^^^^^193^214^return true;^[CLASS] AbstractPieItemLabelGenerator  [METHOD] equals [RETURN_TYPE] boolean   Object obj [VARIABLES] AbstractPieItemLabelGenerator  that  Object  obj  String  labelFormat  result  boolean  long  serialVersionUID  NumberFormat  numberFormat  percentFormat  
[BugLab_Wrong_Operator]^if  ( ! ( obj  >=  AbstractPieItemLabelGenerator )  )  {^197^^^^^193^214^if  ( ! ( obj instanceof AbstractPieItemLabelGenerator )  )  {^[CLASS] AbstractPieItemLabelGenerator  [METHOD] equals [RETURN_TYPE] boolean   Object obj [VARIABLES] AbstractPieItemLabelGenerator  that  Object  obj  String  labelFormat  result  boolean  long  serialVersionUID  NumberFormat  numberFormat  percentFormat  
[BugLab_Wrong_Literal]^return true;^198^^^^^193^214^return false;^[CLASS] AbstractPieItemLabelGenerator  [METHOD] equals [RETURN_TYPE] boolean   Object obj [VARIABLES] AbstractPieItemLabelGenerator  that  Object  obj  String  labelFormat  result  boolean  long  serialVersionUID  NumberFormat  numberFormat  percentFormat  
[BugLab_Variable_Misuse]^if  ( !this.labelFormat.equals ( result )  )  {^203^^^^^193^214^if  ( !this.labelFormat.equals ( that.labelFormat )  )  {^[CLASS] AbstractPieItemLabelGenerator  [METHOD] equals [RETURN_TYPE] boolean   Object obj [VARIABLES] AbstractPieItemLabelGenerator  that  Object  obj  String  labelFormat  result  boolean  long  serialVersionUID  NumberFormat  numberFormat  percentFormat  
[BugLab_Wrong_Literal]^return true;^204^^^^^193^214^return false;^[CLASS] AbstractPieItemLabelGenerator  [METHOD] equals [RETURN_TYPE] boolean   Object obj [VARIABLES] AbstractPieItemLabelGenerator  that  Object  obj  String  labelFormat  result  boolean  long  serialVersionUID  NumberFormat  numberFormat  percentFormat  
[BugLab_Variable_Misuse]^if  ( !this.numberFormat.equals ( percentFormat )  )  {^206^^^^^193^214^if  ( !this.numberFormat.equals ( that.numberFormat )  )  {^[CLASS] AbstractPieItemLabelGenerator  [METHOD] equals [RETURN_TYPE] boolean   Object obj [VARIABLES] AbstractPieItemLabelGenerator  that  Object  obj  String  labelFormat  result  boolean  long  serialVersionUID  NumberFormat  numberFormat  percentFormat  
[BugLab_Argument_Swapping]^if  ( !this.numberFormat.equals ( that.numberFormat.numberFormat )  )  {^206^^^^^193^214^if  ( !this.numberFormat.equals ( that.numberFormat )  )  {^[CLASS] AbstractPieItemLabelGenerator  [METHOD] equals [RETURN_TYPE] boolean   Object obj [VARIABLES] AbstractPieItemLabelGenerator  that  Object  obj  String  labelFormat  result  boolean  long  serialVersionUID  NumberFormat  numberFormat  percentFormat  
[BugLab_Wrong_Literal]^return true;^207^^^^^193^214^return false;^[CLASS] AbstractPieItemLabelGenerator  [METHOD] equals [RETURN_TYPE] boolean   Object obj [VARIABLES] AbstractPieItemLabelGenerator  that  Object  obj  String  labelFormat  result  boolean  long  serialVersionUID  NumberFormat  numberFormat  percentFormat  
[BugLab_Argument_Swapping]^if  ( !this.percentFormat.equals ( that )  )  {^209^^^^^193^214^if  ( !this.percentFormat.equals ( that.percentFormat )  )  {^[CLASS] AbstractPieItemLabelGenerator  [METHOD] equals [RETURN_TYPE] boolean   Object obj [VARIABLES] AbstractPieItemLabelGenerator  that  Object  obj  String  labelFormat  result  boolean  long  serialVersionUID  NumberFormat  numberFormat  percentFormat  
[BugLab_Wrong_Literal]^return true;^210^^^^^193^214^return false;^[CLASS] AbstractPieItemLabelGenerator  [METHOD] equals [RETURN_TYPE] boolean   Object obj [VARIABLES] AbstractPieItemLabelGenerator  that  Object  obj  String  labelFormat  result  boolean  long  serialVersionUID  NumberFormat  numberFormat  percentFormat  
[BugLab_Variable_Misuse]^if  ( !this.percentFormat.equals ( percentFormat )  )  {^209^^^^^193^214^if  ( !this.percentFormat.equals ( that.percentFormat )  )  {^[CLASS] AbstractPieItemLabelGenerator  [METHOD] equals [RETURN_TYPE] boolean   Object obj [VARIABLES] AbstractPieItemLabelGenerator  that  Object  obj  String  labelFormat  result  boolean  long  serialVersionUID  NumberFormat  numberFormat  percentFormat  
[BugLab_Wrong_Literal]^return false;^212^^^^^193^214^return true;^[CLASS] AbstractPieItemLabelGenerator  [METHOD] equals [RETURN_TYPE] boolean   Object obj [VARIABLES] AbstractPieItemLabelGenerator  that  Object  obj  String  labelFormat  result  boolean  long  serialVersionUID  NumberFormat  numberFormat  percentFormat  
[BugLab_Variable_Misuse]^if  ( percentFormat != null )  {^226^^^^^223^233^if  ( this.numberFormat != null )  {^[CLASS] AbstractPieItemLabelGenerator  [METHOD] clone [RETURN_TYPE] Object   [VARIABLES] AbstractPieItemLabelGenerator  clone  String  labelFormat  result  boolean  long  serialVersionUID  NumberFormat  numberFormat  percentFormat  
[BugLab_Wrong_Operator]^if  ( this.numberFormat == null )  {^226^^^^^223^233^if  ( this.numberFormat != null )  {^[CLASS] AbstractPieItemLabelGenerator  [METHOD] clone [RETURN_TYPE] Object   [VARIABLES] AbstractPieItemLabelGenerator  clone  String  labelFormat  result  boolean  long  serialVersionUID  NumberFormat  numberFormat  percentFormat  
[BugLab_Variable_Misuse]^clone.numberFormat =  ( NumberFormat )  percentFormat.clone (  ) ;^227^^^^^223^233^clone.numberFormat =  ( NumberFormat )  this.numberFormat.clone (  ) ;^[CLASS] AbstractPieItemLabelGenerator  [METHOD] clone [RETURN_TYPE] Object   [VARIABLES] AbstractPieItemLabelGenerator  clone  String  labelFormat  result  boolean  long  serialVersionUID  NumberFormat  numberFormat  percentFormat  
[BugLab_Wrong_Operator]^if  ( this.percentFormat == null )  {^229^^^^^223^233^if  ( this.percentFormat != null )  {^[CLASS] AbstractPieItemLabelGenerator  [METHOD] clone [RETURN_TYPE] Object   [VARIABLES] AbstractPieItemLabelGenerator  clone  String  labelFormat  result  boolean  long  serialVersionUID  NumberFormat  numberFormat  percentFormat  
[BugLab_Variable_Misuse]^clone.percentFormat =  ( NumberFormat )  percentFormat.clone (  ) ;^230^^^^^223^233^clone.percentFormat =  ( NumberFormat )  this.percentFormat.clone (  ) ;^[CLASS] AbstractPieItemLabelGenerator  [METHOD] clone [RETURN_TYPE] Object   [VARIABLES] AbstractPieItemLabelGenerator  clone  String  labelFormat  result  boolean  long  serialVersionUID  NumberFormat  numberFormat  percentFormat  
