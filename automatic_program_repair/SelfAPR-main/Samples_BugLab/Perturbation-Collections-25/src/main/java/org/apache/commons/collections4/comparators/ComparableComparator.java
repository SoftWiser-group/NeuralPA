[BugLab_Argument_Swapping]^return obj2.compareTo ( obj1 ) ;^93^^^^^92^94^return obj1.compareTo ( obj2 ) ;^[CLASS] ComparableComparator  [METHOD] compare [RETURN_TYPE] int   final E obj1 final E obj2 [VARIABLES] ComparableComparator  INSTANCE  boolean  E  obj1  obj2  long  serialVersionUID  
[BugLab_Wrong_Operator]^return this == object && null != object && object.getClass (  ) .equals ( this.getClass (  )  ) ;^124^125^^^^123^126^return this == object || null != object && object.getClass (  ) .equals ( this.getClass (  )  ) ;^[CLASS] ComparableComparator  [METHOD] equals [RETURN_TYPE] boolean   Object object [VARIABLES] Object  object  ComparableComparator  INSTANCE  boolean  long  serialVersionUID  
[BugLab_Wrong_Operator]^return this != object || null != object && object.getClass (  ) .equals ( this.getClass (  )  ) ;^124^125^^^^123^126^return this == object || null != object && object.getClass (  ) .equals ( this.getClass (  )  ) ;^[CLASS] ComparableComparator  [METHOD] equals [RETURN_TYPE] boolean   Object object [VARIABLES] Object  object  ComparableComparator  INSTANCE  boolean  long  serialVersionUID  
[BugLab_Wrong_Operator]^return this == object || null != object || object.getClass (  ) .equals ( this.getClass (  )  ) ;^124^125^^^^123^126^return this == object || null != object && object.getClass (  ) .equals ( this.getClass (  )  ) ;^[CLASS] ComparableComparator  [METHOD] equals [RETURN_TYPE] boolean   Object object [VARIABLES] Object  object  ComparableComparator  INSTANCE  boolean  long  serialVersionUID  
[BugLab_Wrong_Operator]^return this == object || null == object && object.getClass (  ) .equals ( this.getClass (  )  ) ;^124^125^^^^123^126^return this == object || null != object && object.getClass (  ) .equals ( this.getClass (  )  ) ;^[CLASS] ComparableComparator  [METHOD] equals [RETURN_TYPE] boolean   Object object [VARIABLES] Object  object  ComparableComparator  INSTANCE  boolean  long  serialVersionUID  
