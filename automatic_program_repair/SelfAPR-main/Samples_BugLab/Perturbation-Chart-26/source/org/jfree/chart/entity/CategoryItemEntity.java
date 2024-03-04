[BugLab_Variable_Misuse]^super ( area, urlText, urlText ) ;^104^^^^^102^111^super ( area, toolTipText, urlText ) ;^[CLASS] CategoryItemEntity  [METHOD] <init> [RETURN_TYPE] Comparable)   Shape area String toolTipText String urlText CategoryDataset dataset Comparable rowKey Comparable columnKey [VARIABLES] Comparable  columnKey  rowKey  Shape  area  String  toolTipText  urlText  boolean  long  serialVersionUID  CategoryDataset  dataset  
[BugLab_Argument_Swapping]^super ( urlText, toolTipText, area ) ;^104^^^^^102^111^super ( area, toolTipText, urlText ) ;^[CLASS] CategoryItemEntity  [METHOD] <init> [RETURN_TYPE] Comparable)   Shape area String toolTipText String urlText CategoryDataset dataset Comparable rowKey Comparable columnKey [VARIABLES] Comparable  columnKey  rowKey  Shape  area  String  toolTipText  urlText  boolean  long  serialVersionUID  CategoryDataset  dataset  
[BugLab_Argument_Swapping]^super ( area, urlText, toolTipText ) ;^104^^^^^102^111^super ( area, toolTipText, urlText ) ;^[CLASS] CategoryItemEntity  [METHOD] <init> [RETURN_TYPE] Comparable)   Shape area String toolTipText String urlText CategoryDataset dataset Comparable rowKey Comparable columnKey [VARIABLES] Comparable  columnKey  rowKey  Shape  area  String  toolTipText  urlText  boolean  long  serialVersionUID  CategoryDataset  dataset  
[BugLab_Wrong_Operator]^if  ( dataset != null )  {^105^^^^^102^111^if  ( dataset == null )  {^[CLASS] CategoryItemEntity  [METHOD] <init> [RETURN_TYPE] Comparable)   Shape area String toolTipText String urlText CategoryDataset dataset Comparable rowKey Comparable columnKey [VARIABLES] Comparable  columnKey  rowKey  Shape  area  String  toolTipText  urlText  boolean  long  serialVersionUID  CategoryDataset  dataset  
[BugLab_Variable_Misuse]^this.rowKey = columnKey;^109^^^^^102^111^this.rowKey = rowKey;^[CLASS] CategoryItemEntity  [METHOD] <init> [RETURN_TYPE] Comparable)   Shape area String toolTipText String urlText CategoryDataset dataset Comparable rowKey Comparable columnKey [VARIABLES] Comparable  columnKey  rowKey  Shape  area  String  toolTipText  urlText  boolean  long  serialVersionUID  CategoryDataset  dataset  
[BugLab_Variable_Misuse]^this.columnKey = rowKey;^110^^^^^102^111^this.columnKey = columnKey;^[CLASS] CategoryItemEntity  [METHOD] <init> [RETURN_TYPE] Comparable)   Shape area String toolTipText String urlText CategoryDataset dataset Comparable rowKey Comparable columnKey [VARIABLES] Comparable  columnKey  rowKey  Shape  area  String  toolTipText  urlText  boolean  long  serialVersionUID  CategoryDataset  dataset  
[BugLab_Variable_Misuse]^return dataset;^123^^^^^122^124^return this.dataset;^[CLASS] CategoryItemEntity  [METHOD] getDataset [RETURN_TYPE] CategoryDataset   [VARIABLES] Comparable  columnKey  rowKey  boolean  long  serialVersionUID  CategoryDataset  dataset  
[BugLab_Wrong_Operator]^if  ( dataset != null )  {^134^^^^^133^138^if  ( dataset == null )  {^[CLASS] CategoryItemEntity  [METHOD] setDataset [RETURN_TYPE] void   CategoryDataset dataset [VARIABLES] Comparable  columnKey  rowKey  boolean  long  serialVersionUID  CategoryDataset  dataset  
[BugLab_Variable_Misuse]^return rowKey;^150^^^^^149^151^return this.rowKey;^[CLASS] CategoryItemEntity  [METHOD] getRowKey [RETURN_TYPE] Comparable   [VARIABLES] Comparable  columnKey  rowKey  boolean  long  serialVersionUID  CategoryDataset  dataset  
[BugLab_Variable_Misuse]^this.rowKey = columnKey;^163^^^^^162^164^this.rowKey = rowKey;^[CLASS] CategoryItemEntity  [METHOD] setRowKey [RETURN_TYPE] void   Comparable rowKey [VARIABLES] Comparable  columnKey  rowKey  boolean  long  serialVersionUID  CategoryDataset  dataset  
[BugLab_Variable_Misuse]^this.columnKey = rowKey;^189^^^^^188^190^this.columnKey = columnKey;^[CLASS] CategoryItemEntity  [METHOD] setColumnKey [RETURN_TYPE] void   Comparable columnKey [VARIABLES] Comparable  columnKey  rowKey  boolean  long  serialVersionUID  CategoryDataset  dataset  
[BugLab_Variable_Misuse]^return "CategoryItemEntity: rowKey=" + rowKey + ", columnKey=" + this.columnKey + ", dataset=" + this.dataset;^199^200^^^^198^201^return "CategoryItemEntity: rowKey=" + this.rowKey + ", columnKey=" + this.columnKey + ", dataset=" + this.dataset;^[CLASS] CategoryItemEntity  [METHOD] toString [RETURN_TYPE] String   [VARIABLES] Comparable  columnKey  rowKey  boolean  long  serialVersionUID  CategoryDataset  dataset  
[BugLab_Variable_Misuse]^return "CategoryItemEntity: rowKey=" + this.rowKey + ", columnKey=" + rowKey + ", dataset=" + this.dataset;^199^200^^^^198^201^return "CategoryItemEntity: rowKey=" + this.rowKey + ", columnKey=" + this.columnKey + ", dataset=" + this.dataset;^[CLASS] CategoryItemEntity  [METHOD] toString [RETURN_TYPE] String   [VARIABLES] Comparable  columnKey  rowKey  boolean  long  serialVersionUID  CategoryDataset  dataset  
[BugLab_Variable_Misuse]^return "CategoryItemEntity: rowKey=" + this.rowKey + ", columnKey=" + this.columnKey + ", dataset=" + dataset;^199^200^^^^198^201^return "CategoryItemEntity: rowKey=" + this.rowKey + ", columnKey=" + this.columnKey + ", dataset=" + this.dataset;^[CLASS] CategoryItemEntity  [METHOD] toString [RETURN_TYPE] String   [VARIABLES] Comparable  columnKey  rowKey  boolean  long  serialVersionUID  CategoryDataset  dataset  
[BugLab_Argument_Swapping]^return "CategoryItemEntity: rowKey=" + this.columnKey + ", columnKey=" + this.rowKey + ", dataset=" + this.dataset;^199^200^^^^198^201^return "CategoryItemEntity: rowKey=" + this.rowKey + ", columnKey=" + this.columnKey + ", dataset=" + this.dataset;^[CLASS] CategoryItemEntity  [METHOD] toString [RETURN_TYPE] String   [VARIABLES] Comparable  columnKey  rowKey  boolean  long  serialVersionUID  CategoryDataset  dataset  
[BugLab_Argument_Swapping]^return "CategoryItemEntity: rowKey=" + this.dataset + ", columnKey=" + this.columnKey + ", dataset=" + this.rowKey;^199^200^^^^198^201^return "CategoryItemEntity: rowKey=" + this.rowKey + ", columnKey=" + this.columnKey + ", dataset=" + this.dataset;^[CLASS] CategoryItemEntity  [METHOD] toString [RETURN_TYPE] String   [VARIABLES] Comparable  columnKey  rowKey  boolean  long  serialVersionUID  CategoryDataset  dataset  
[BugLab_Wrong_Operator]^return "CategoryItemEntity: rowKey="  <  this.rowKey + ", columnKey=" + this.columnKey + ", dataset=" + this.dataset;^199^200^^^^198^201^return "CategoryItemEntity: rowKey=" + this.rowKey + ", columnKey=" + this.columnKey + ", dataset=" + this.dataset;^[CLASS] CategoryItemEntity  [METHOD] toString [RETURN_TYPE] String   [VARIABLES] Comparable  columnKey  rowKey  boolean  long  serialVersionUID  CategoryDataset  dataset  
[BugLab_Wrong_Operator]^if  ( obj > this )  {^211^^^^^210^228^if  ( obj == this )  {^[CLASS] CategoryItemEntity  [METHOD] equals [RETURN_TYPE] boolean   Object obj [VARIABLES] Comparable  columnKey  rowKey  Object  obj  boolean  CategoryItemEntity  that  long  serialVersionUID  CategoryDataset  dataset  
[BugLab_Wrong_Literal]^return false;^212^^^^^210^228^return true;^[CLASS] CategoryItemEntity  [METHOD] equals [RETURN_TYPE] boolean   Object obj [VARIABLES] Comparable  columnKey  rowKey  Object  obj  boolean  CategoryItemEntity  that  long  serialVersionUID  CategoryDataset  dataset  
[BugLab_Wrong_Operator]^if  ( ! ( obj  ||  CategoryItemEntity )  )  {^214^^^^^210^228^if  ( ! ( obj instanceof CategoryItemEntity )  )  {^[CLASS] CategoryItemEntity  [METHOD] equals [RETURN_TYPE] boolean   Object obj [VARIABLES] Comparable  columnKey  rowKey  Object  obj  boolean  CategoryItemEntity  that  long  serialVersionUID  CategoryDataset  dataset  
[BugLab_Wrong_Literal]^return true;^215^^^^^210^228^return false;^[CLASS] CategoryItemEntity  [METHOD] equals [RETURN_TYPE] boolean   Object obj [VARIABLES] Comparable  columnKey  rowKey  Object  obj  boolean  CategoryItemEntity  that  long  serialVersionUID  CategoryDataset  dataset  
[BugLab_Variable_Misuse]^if  ( !this.rowKey.equals ( rowKey )  )  {^218^^^^^210^228^if  ( !this.rowKey.equals ( that.rowKey )  )  {^[CLASS] CategoryItemEntity  [METHOD] equals [RETURN_TYPE] boolean   Object obj [VARIABLES] Comparable  columnKey  rowKey  Object  obj  boolean  CategoryItemEntity  that  long  serialVersionUID  CategoryDataset  dataset  
[BugLab_Argument_Swapping]^if  ( !this.rowKey.equals ( that.rowKey.rowKey )  )  {^218^^^^^210^228^if  ( !this.rowKey.equals ( that.rowKey )  )  {^[CLASS] CategoryItemEntity  [METHOD] equals [RETURN_TYPE] boolean   Object obj [VARIABLES] Comparable  columnKey  rowKey  Object  obj  boolean  CategoryItemEntity  that  long  serialVersionUID  CategoryDataset  dataset  
[BugLab_Wrong_Literal]^return true;^219^^^^^210^228^return false;^[CLASS] CategoryItemEntity  [METHOD] equals [RETURN_TYPE] boolean   Object obj [VARIABLES] Comparable  columnKey  rowKey  Object  obj  boolean  CategoryItemEntity  that  long  serialVersionUID  CategoryDataset  dataset  
[BugLab_Variable_Misuse]^if  ( !this.columnKey.equals ( rowKey )  )  {^221^^^^^210^228^if  ( !this.columnKey.equals ( that.columnKey )  )  {^[CLASS] CategoryItemEntity  [METHOD] equals [RETURN_TYPE] boolean   Object obj [VARIABLES] Comparable  columnKey  rowKey  Object  obj  boolean  CategoryItemEntity  that  long  serialVersionUID  CategoryDataset  dataset  
[BugLab_Argument_Swapping]^if  ( !this.columnKey.equals ( that.columnKey.columnKey )  )  {^221^^^^^210^228^if  ( !this.columnKey.equals ( that.columnKey )  )  {^[CLASS] CategoryItemEntity  [METHOD] equals [RETURN_TYPE] boolean   Object obj [VARIABLES] Comparable  columnKey  rowKey  Object  obj  boolean  CategoryItemEntity  that  long  serialVersionUID  CategoryDataset  dataset  
[BugLab_Wrong_Literal]^return true;^222^^^^^210^228^return false;^[CLASS] CategoryItemEntity  [METHOD] equals [RETURN_TYPE] boolean   Object obj [VARIABLES] Comparable  columnKey  rowKey  Object  obj  boolean  CategoryItemEntity  that  long  serialVersionUID  CategoryDataset  dataset  
[BugLab_Variable_Misuse]^if  ( !ObjectUtilities.equal ( dataset, that.dataset )  )  {^224^^^^^210^228^if  ( !ObjectUtilities.equal ( this.dataset, that.dataset )  )  {^[CLASS] CategoryItemEntity  [METHOD] equals [RETURN_TYPE] boolean   Object obj [VARIABLES] Comparable  columnKey  rowKey  Object  obj  boolean  CategoryItemEntity  that  long  serialVersionUID  CategoryDataset  dataset  
[BugLab_Variable_Misuse]^if  ( !ObjectUtilities.equal ( this.dataset, dataset )  )  {^224^^^^^210^228^if  ( !ObjectUtilities.equal ( this.dataset, that.dataset )  )  {^[CLASS] CategoryItemEntity  [METHOD] equals [RETURN_TYPE] boolean   Object obj [VARIABLES] Comparable  columnKey  rowKey  Object  obj  boolean  CategoryItemEntity  that  long  serialVersionUID  CategoryDataset  dataset  
[BugLab_Argument_Swapping]^if  ( !ObjectUtilities.equal ( this.dataset, that.dataset.dataset )  )  {^224^^^^^210^228^if  ( !ObjectUtilities.equal ( this.dataset, that.dataset )  )  {^[CLASS] CategoryItemEntity  [METHOD] equals [RETURN_TYPE] boolean   Object obj [VARIABLES] Comparable  columnKey  rowKey  Object  obj  boolean  CategoryItemEntity  that  long  serialVersionUID  CategoryDataset  dataset  
[BugLab_Argument_Swapping]^if  ( !ObjectUtilities.equal ( that.dataset, this.dataset )  )  {^224^^^^^210^228^if  ( !ObjectUtilities.equal ( this.dataset, that.dataset )  )  {^[CLASS] CategoryItemEntity  [METHOD] equals [RETURN_TYPE] boolean   Object obj [VARIABLES] Comparable  columnKey  rowKey  Object  obj  boolean  CategoryItemEntity  that  long  serialVersionUID  CategoryDataset  dataset  
[BugLab_Wrong_Literal]^return true;^225^^^^^210^228^return false;^[CLASS] CategoryItemEntity  [METHOD] equals [RETURN_TYPE] boolean   Object obj [VARIABLES] Comparable  columnKey  rowKey  Object  obj  boolean  CategoryItemEntity  that  long  serialVersionUID  CategoryDataset  dataset  
